#!/usr/bin/env python3
"""Franka real-robot client for OpenPI pi05 with deployable realtime chunking.

This script is meant to run in the ``franka_teleop`` conda environment while
an OpenPI websocket policy server runs in the ``openpi`` environment.
"""

from __future__ import annotations

import argparse
import math
import pickle
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import zmq
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from polymetis import GripperInterface, RobotInterface

MAX_OPEN_M = 0.09
JOINT_DIM = 7
ACTION_DIM = 8
IMAGE_SIZE = 224

# Franka Panda/FR3 conservative hard limits.
JOINT_LIMIT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
JOINT_LIMIT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)


class ZMQClientCamera:
    """REQ client for teleop ZMQ camera nodes."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self.host = host
        self.port = port
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        self._socket.send(pickle.dumps(None))
        image, depth = pickle.loads(self._socket.recv())
        return image, depth


@dataclass(frozen=True)
class CameraSetup:
    exterior: ZMQClientCamera
    wrist: ZMQClientCamera | None
    detected_ports: tuple[int, ...]


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = True
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    schedule: Literal["exp", "linear", "ones", "zeros"] = "exp"

    def blend_weight(self, index: int, inference_delay: int) -> float:
        if not self.enabled:
            return 0.0
        if self.schedule == "ones":
            base = 1.0
        elif self.schedule == "zeros":
            base = 1.0 if index < inference_delay else 0.0
        elif self.schedule == "linear":
            denom = max(self.execution_horizon - 1, 1)
            base = max(0.0, 1.0 - index / denom)
        else:
            tau = max(float(self.execution_horizon) / 3.0, 1e-6)
            base = float(math.exp(-index / tau))

        scale = float(np.clip(self.max_guidance_weight / 10.0, 0.0, 1.0))
        return float(np.clip(base * scale, 0.0, 1.0))


@dataclass(frozen=True)
class RuntimeStats:
    infer_latency_ms: float | None = None
    inferred_delay_steps: int = 0
    dropped_stale_steps: int = 0
    merged_overlap_steps: int = 0


class RealtimeActionQueue:
    """FIFO queue with delay compensation and RTC-style prefix blending."""

    def __init__(self, rtc: RTCConfig, max_queue_size: int):
        self._rtc = rtc
        self._max_queue_size = max_queue_size
        self._queue: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stats = RuntimeStats()

    def qsize(self) -> int:
        with self._lock:
            return len(self._queue)

    def get(self) -> np.ndarray | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft().copy()

    def last_stats(self) -> RuntimeStats:
        with self._lock:
            return self._stats

    def merge_chunk(self, chunk: np.ndarray, inference_delay: int, infer_latency_ms: float) -> None:
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise ValueError(f"Unexpected action chunk shape: {chunk.shape}")

        stale_steps = min(max(inference_delay, 0), len(chunk))
        fresh_chunk = chunk[stale_steps:].copy()
        if len(fresh_chunk) == 0:
            with self._lock:
                self._stats = RuntimeStats(
                    infer_latency_ms=infer_latency_ms,
                    inferred_delay_steps=inference_delay,
                    dropped_stale_steps=stale_steps,
                    merged_overlap_steps=0,
                )
            return

        with self._lock:
            current = list(self._queue)
            overlap = 0
            if self._rtc.enabled and current:
                overlap = min(self._rtc.execution_horizon, len(current), len(fresh_chunk))
                for idx in range(overlap):
                    w = self._rtc.blend_weight(idx, inference_delay)
                    fresh_chunk[idx] = w * current[idx] + (1.0 - w) * fresh_chunk[idx]
                current[:overlap] = [step.copy() for step in fresh_chunk[:overlap]]
                current.extend(step.copy() for step in fresh_chunk[overlap:])
            else:
                current.extend(step.copy() for step in fresh_chunk)

            if len(current) > self._max_queue_size:
                current = current[: self._max_queue_size]

            self._queue = deque(current)
            self._stats = RuntimeStats(
                infer_latency_ms=infer_latency_ms,
                inferred_delay_steps=inference_delay,
                dropped_stale_steps=stale_steps,
                merged_overlap_steps=overlap,
            )


class FrankaRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.stop_event = threading.Event()
        self.errors: "queue.Queue[BaseException]" = queue.Queue()

        self.policy = websocket_client_policy.WebsocketClientPolicy(args.policy_host, args.policy_port)
        self.robot = RobotInterface(ip_address=args.robot_ip, port=args.franka_port)
        self.gripper = GripperInterface(ip_address=args.robot_ip, port=args.gripper_port)
        self.cameras = self._initialize_cameras()
        self.chunk_queue = RealtimeActionQueue(
            rtc=RTCConfig(
                enabled=not args.disable_rtc_style,
                execution_horizon=args.execution_horizon,
                max_guidance_weight=args.max_guidance_weight,
                schedule=args.rtc_schedule,
            ),
            max_queue_size=args.max_queue_size,
        )

        self.last_gripper_cmd = -1.0
        self.last_gripper_time = 0.0
        self.last_sent_joint_cmd: np.ndarray | None = None
        self.last_joint_delta = np.zeros(JOINT_DIM, dtype=np.float32)
        self.last_executed_action: np.ndarray | None = None
        self.estimated_inference_delay = max(0.0, args.initial_inference_delay)

    def _candidate_camera_ports(self) -> list[int]:
        explicit = []
        if self.args.exterior_camera_port >= 0:
            explicit.append(self.args.exterior_camera_port)
        if self.args.wrist_camera_port >= 0:
            explicit.append(self.args.wrist_camera_port)
        if explicit:
            return list(dict.fromkeys(explicit))
        return [self.args.camera_port_base + offset for offset in range(self.args.camera_max_ports)]

    def _probe_camera(self, port: int) -> tuple[ZMQClientCamera | None, tuple[int, int, int] | None]:
        try:
            camera = ZMQClientCamera(self.args.camera_host, port, self.args.camera_timeout_ms)
            image_bgr, _ = camera.read()
            if image_bgr is None or image_bgr.ndim != 3:
                raise RuntimeError(f"unexpected image from camera port {port}: {type(image_bgr)}")
            shape = tuple(int(x) for x in image_bgr.shape)
            return camera, shape
        except Exception as exc:  # noqa: BLE001
            print(f"[camera] skip port={port} reason={exc}")
            return None, None

    def _initialize_cameras(self) -> CameraSetup:
        candidates = self._candidate_camera_ports()
        found: list[tuple[int, ZMQClientCamera, tuple[int, int, int]]] = []

        print(f"[camera] probing host={self.args.camera_host} candidate_ports={candidates}")
        for port in candidates:
            camera, shape = self._probe_camera(port)
            if camera is None or shape is None:
                continue
            found.append((port, camera, shape))
            print(f"[camera] detected port={port} frame_shape={shape}")

        if not found:
            raise RuntimeError(
                f"No camera detected on host={self.args.camera_host}; tried ports={candidates}"
            )

        if len(found) == 1:
            exterior_port, exterior_cam, shape = found[0]
            print(
                f"[camera] connected 1 camera: exterior=port {exterior_port}, wrist=disabled, frame_shape={shape}"
            )
            return CameraSetup(exterior=exterior_cam, wrist=None, detected_ports=(exterior_port,))

        exterior_port, exterior_cam, exterior_shape = found[0]
        wrist_port, wrist_cam, wrist_shape = found[1]
        print(
            f"[camera] connected {len(found)} cameras; use exterior=port {exterior_port} shape={exterior_shape}, "
            f"wrist=port {wrist_port} shape={wrist_shape}"
        )
        if len(found) > 2:
            extra_ports = [port for port, _, _ in found[2:]]
            print(f"[camera] extra detected cameras ignored: {extra_ports}")
        return CameraSetup(exterior=exterior_cam, wrist=wrist_cam, detected_ports=tuple(port for port, _, _ in found))

    def _observe(self) -> dict:
        ext_img_bgr, _ = self.cameras.exterior.read()
        ext_img = image_tools.resize_with_pad(_bgr_to_rgb(ext_img_bgr), IMAGE_SIZE, IMAGE_SIZE)

        if self.cameras.wrist is not None:
            wrist_img_bgr, _ = self.cameras.wrist.read()
            wrist_img = image_tools.resize_with_pad(_bgr_to_rgb(wrist_img_bgr), IMAGE_SIZE, IMAGE_SIZE)
        else:
            wrist_img = np.zeros_like(ext_img)

        q = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
        gripper_width = float(self.gripper.get_state().width)
        gripper_closed = np.array([1.0 - np.clip(gripper_width / MAX_OPEN_M, 0.0, 1.0)], dtype=np.float32)

        return {
            "observation/exterior_image_1_left": ext_img,
            "observation/wrist_image_left": wrist_img,
            "observation/joint_position": q,
            "observation/gripper_position": gripper_closed,
            "prompt": self.args.prompt,
        }

    def _infer_loop(self) -> None:
        try:
            control_dt = 1.0 / self.args.control_hz
            while not self.stop_event.is_set():
                if self.chunk_queue.qsize() > self.args.queue_refill_threshold:
                    time.sleep(0.002)
                    continue

                obs = self._observe()
                t0 = time.perf_counter()
                result = self.policy.infer(obs)
                infer_s = time.perf_counter() - t0
                raw_actions = np.asarray(result["actions"], dtype=np.float32)
                actions = self._sanitize_actions(raw_actions)

                measured_delay = infer_s / control_dt
                self.estimated_inference_delay = (
                    self.args.inference_delay_ema_decay * self.estimated_inference_delay
                    + (1.0 - self.args.inference_delay_ema_decay) * measured_delay
                )
                inference_delay = int(max(0, round(self.estimated_inference_delay + self.args.extra_delay_steps)))

                self.chunk_queue.merge_chunk(
                    actions,
                    inference_delay=inference_delay,
                    infer_latency_ms=infer_s * 1000.0,
                )
        except BaseException as exc:  # noqa: BLE001
            self.errors.put(exc)
            self.stop_event.set()

    def _sanitize_actions(self, actions: np.ndarray) -> np.ndarray:
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
            raise ValueError(f"Unexpected action shape: {actions.shape}, expected [T, >=8]")

        actions = actions[:, :ACTION_DIM].astype(np.float32, copy=True)
        actions[:, :JOINT_DIM] = np.clip(actions[:, :JOINT_DIM], JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)
        actions[:, 7] = np.clip(actions[:, 7], 0.0, 1.0)

        for idx in range(1, len(actions)):
            delta = np.clip(
                actions[idx, :JOINT_DIM] - actions[idx - 1, :JOINT_DIM],
                -self.args.max_joint_step_rad,
                self.args.max_joint_step_rad,
            )
            actions[idx, :JOINT_DIM] = actions[idx - 1, :JOINT_DIM] + delta
        return actions

    def _compute_joint_command(self, action: np.ndarray) -> np.ndarray:
        q_cur = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
        q_target = np.clip(action[:JOINT_DIM], JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)

        raw_delta = q_target - q_cur
        delta = np.clip(raw_delta, -self.args.max_joint_step_rad, self.args.max_joint_step_rad)

        accel_limit = self.args.max_delta_change_rad
        delta = np.clip(delta, self.last_joint_delta - accel_limit, self.last_joint_delta + accel_limit)

        q_cmd = np.clip(q_cur + delta, JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)
        if self.last_sent_joint_cmd is not None and self.args.joint_command_alpha < 1.0:
            q_cmd = (
                self.args.joint_command_alpha * q_cmd
                + (1.0 - self.args.joint_command_alpha) * self.last_sent_joint_cmd
            ).astype(np.float32)

        self.last_joint_delta = q_cmd - q_cur
        self.last_sent_joint_cmd = q_cmd.copy()
        return q_cmd

    def _execute_action(self, action: np.ndarray) -> None:
        q_cmd = self._compute_joint_command(action)
        self.robot.update_desired_joint_positions(torch.from_numpy(q_cmd).float())

        g_tgt = float(np.clip(action[7], 0.0, 1.0))
        if self.args.binarize_gripper:
            g_tgt = 1.0 if g_tgt >= self.args.gripper_binary_threshold else 0.0

        now = time.time()
        if (
            abs(g_tgt - self.last_gripper_cmd) >= self.args.gripper_deadband
            and (now - self.last_gripper_time) >= self.args.gripper_min_interval_s
        ):
            width = float((1.0 - g_tgt) * MAX_OPEN_M)
            self.gripper.goto(width=width, speed=self.args.gripper_speed, force=self.args.gripper_force)
            self.last_gripper_cmd = g_tgt
            self.last_gripper_time = now

        self.last_executed_action = action.copy()

    def _get_fallback_action(self) -> np.ndarray | None:
        if self.last_executed_action is not None:
            return self.last_executed_action.copy()
        if not self.args.hold_position_on_empty_queue:
            return None

        q_cur = self.robot.get_joint_positions().detach().cpu().numpy().astype(np.float32)
        gripper_width = float(self.gripper.get_state().width)
        g_cur = 1.0 - np.clip(gripper_width / MAX_OPEN_M, 0.0, 1.0)
        action = np.concatenate([q_cur, np.array([g_cur], dtype=np.float32)])
        return action.astype(np.float32)

    def run(self) -> None:
        if self.args.go_home:
            print("[init] robot.go_home()")
            self.robot.go_home()

        print("[init] start_joint_impedance()")
        self.robot.start_joint_impedance()

        if self.args.open_gripper_on_start:
            self.gripper.goto(width=MAX_OPEN_M, speed=self.args.gripper_speed, force=self.args.gripper_force)
            time.sleep(0.5)

        print("[init] warmup infer...")
        warmup_actions = self._sanitize_actions(np.asarray(self.policy.infer(self._observe())["actions"], dtype=np.float32))
        self.chunk_queue.merge_chunk(warmup_actions, inference_delay=0, infer_latency_ms=0.0)
        print("[init] warmup done")

        infer_thread = threading.Thread(target=self._infer_loop, daemon=True, name="infer-thread")
        infer_thread.start()

        step_dt = 1.0 / self.args.control_hz
        t_start = time.time()
        step = 0
        print("[run] started")

        try:
            while not self.stop_event.is_set():
                if self.args.duration_s > 0 and (time.time() - t_start) > self.args.duration_s:
                    break
                if not self.errors.empty():
                    raise self.errors.get()

                loop_t0 = time.perf_counter()
                action = self.chunk_queue.get()
                if action is None:
                    action = self._get_fallback_action()
                if action is not None:
                    self._execute_action(action)

                if step % max(1, int(round(self.args.control_hz))) == 0:
                    stats = self.chunk_queue.last_stats()
                    latency_msg = f"{stats.infer_latency_ms:.1f}ms" if stats.infer_latency_ms is not None else "n/a"
                    print(
                        f"[run] step={step} queue={self.chunk_queue.qsize()} "
                        f"infer={latency_msg} delay_steps={stats.inferred_delay_steps} "
                        f"drop={stats.dropped_stale_steps} overlap={stats.merged_overlap_steps}"
                    )
        
                step += 1
                spent = time.perf_counter() - loop_t0
                time.sleep(max(0.0, step_dt - spent))
        finally:
            self.stop_event.set()
            infer_thread.join(timeout=3.0)
            self.robot.terminate_current_policy()
            print("[done] terminated current robot policy")


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[..., ::-1].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenPI pi05 Franka real-robot client with deployable realtime chunking")

    parser.add_argument("--prompt", type=str, required=True, help="Language instruction for the policy")

    parser.add_argument("--policy_host", type=str, default="127.0.0.1")
    parser.add_argument("--policy_port", type=int, default=8000)

    parser.add_argument("--robot_ip", type=str, default="127.0.0.1")
    parser.add_argument("--franka_port", type=int, default=50051)
    parser.add_argument("--gripper_port", type=int, default=50053)

    parser.add_argument("--camera_host", type=str, default="127.0.0.1")
    parser.add_argument("--camera_port_base", type=int, default=5000)
    parser.add_argument("--camera_max_ports", type=int, default=4)
    parser.add_argument("--exterior_camera_port", type=int, default=-1)
    parser.add_argument("--wrist_camera_port", type=int, default=-1)
    parser.add_argument("--camera_timeout_ms", type=int, default=2000)

    parser.add_argument("--duration_s", type=float, default=120.0, help="<=0 means run until Ctrl+C")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--queue_refill_threshold", type=int, default=4)
    parser.add_argument("--max_queue_size", type=int, default=24)

    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--max_guidance_weight", type=float, default=10.0)
    parser.add_argument("--rtc_schedule", choices=["exp", "linear", "ones", "zeros"], default="exp")
    parser.add_argument("--disable_rtc_style", action="store_true")
    parser.add_argument("--initial_inference_delay", type=float, default=3.0)
    parser.add_argument("--inference_delay_ema_decay", type=float, default=0.7)
    parser.add_argument("--extra_delay_steps", type=float, default=0.0)

    parser.add_argument("--max_joint_step_rad", type=float, default=0.03)
    parser.add_argument("--max_delta_change_rad", type=float, default=0.015)
    parser.add_argument("--joint_command_alpha", type=float, default=0.6)
    parser.add_argument("--hold_position_on_empty_queue", action="store_true")

    parser.add_argument("--binarize_gripper", action="store_true")
    parser.add_argument("--gripper_binary_threshold", type=float, default=0.5)
    parser.add_argument("--gripper_deadband", type=float, default=0.05)
    parser.add_argument("--gripper_min_interval_s", type=float, default=0.2)
    parser.add_argument("--gripper_speed", type=float, default=1.0)
    parser.add_argument("--gripper_force", type=float, default=1.0)

    parser.add_argument("--go_home", action="store_true")
    parser.add_argument("--open_gripper_on_start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = FrankaRuntime(args)

    def _handle_signal(_sig, _frame):
        runtime.stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    runtime.run()


if __name__ == "__main__":
    main()
