"""Convert local Franka teleop pickle episodes to LeRobot format.

The raw data format is produced by /home/server/franka/franka_teleop/teleop/experiments/run_env.py
when launched with --use_save_interface. Each frame is stored as a pickle file containing:

- wrist_rgb / wrist_depth
- base_rgb / base_depth (after enabling base camera capture)
- joint_positions (8D: 7 arm joints + 1 gripper position)
- gripper_position (1D)
- control (8D commanded next joint state / gripper command)

Usage example:
python examples/franka_real/convert_franka_teleop_to_lerobot.py \
  --raw_dir ~/bc_data/toy/original \
  --task "pick up the fork" \
  --swap_wrist_base_labels \
  --output_dir ~/bc_data/toy/franka_singlearm_teleop
"""

from __future__ import annotations

import datetime as dt
import pickle
import shutil
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

REPO_ID = "toy/franka_singlearm_teleop"


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize(size, resample=Image.BICUBIC))


def _load_episode_frames(episode_dir: Path) -> list[tuple[dt.datetime, dict]]:
    frames = []
    for path in sorted(episode_dir.glob('*.pkl')):
        with path.open('rb') as f:
            sample = pickle.load(f)
        stem = path.stem
        try:
            timestamp = dt.datetime.fromisoformat(stem)
        except ValueError:
            if "T" not in stem:
                raise
            date_part, time_part = stem.split("T", 1)
            # Some teleop dumps encode the time portion as HH-MM-SS(.ffffff).
            time_fields = time_part.split("-", 2)
            if len(time_fields) != 3:
                raise
            timestamp = dt.datetime.fromisoformat(f"{date_part}T{':'.join(time_fields)}")
        frames.append((timestamp, sample))
    return frames


def _resample_by_time(frames: list[tuple[dt.datetime, dict]], target_fps: int) -> list[dict]:
    if not frames:
        return []
    interval = dt.timedelta(seconds=1.0 / target_fps)
    next_ts = frames[0][0]
    selected: list[dict] = []
    for ts, sample in frames:
        if ts >= next_ts:
            selected.append(sample)
            next_ts = ts + interval
    return selected


def _extract_frame(
    sample: dict,
    *,
    task: str,
    image_size: tuple[int, int],
    allow_missing_base: bool,
    swap_wrist_base_labels: bool,
) -> dict:
    if 'control' not in sample:
        raise ValueError('Raw frame is missing `control`; this does not look like teleop pickle data.')
    if 'joint_positions' not in sample:
        raise ValueError('Raw frame is missing `joint_positions`.')
    if 'wrist_rgb' not in sample:
        raise ValueError('Raw frame is missing `wrist_rgb`.')

    joint_positions = np.asarray(sample['joint_positions'], dtype=np.float32)
    if joint_positions.shape[-1] < 8:
        raise ValueError(f'Expected 8D joint_positions, got shape {joint_positions.shape}')

    gripper_position = sample.get('gripper_position', joint_positions[7:8])
    gripper_position = np.asarray(gripper_position, dtype=np.float32).reshape(1)
    # Raw teleop observations store normalized opening width: 0=closed, 1=open.
    # OpenPI runtime expects gripper closure: 0=open, 1=closed.
    gripper_position = 1.0 - np.clip(gripper_position, 0.0, 1.0)
    actions = np.asarray(sample['control'], dtype=np.float32)
    if actions.shape[-1] != 8:
        raise ValueError(f'Expected 8D control action, got shape {actions.shape}')
    # Raw teleop gripper actions are already normalized closure commands:
    # 0=open, 1=closed. Keep them unchanged.
    actions = actions.copy()
    actions[7] = np.clip(actions[7], 0.0, 1.0)

    if 'base_rgb' in sample:
        base_rgb = np.asarray(sample['base_rgb'])
    elif allow_missing_base:
        base_rgb = np.zeros_like(np.asarray(sample['wrist_rgb']))
    else:
        raise ValueError('Raw frame is missing `base_rgb`. Re-collect after enabling the base camera or pass --allow-missing-base.')

    wrist_rgb = np.asarray(sample['wrist_rgb'])
    if swap_wrist_base_labels:
        base_rgb, wrist_rgb = wrist_rgb, base_rgb

    width, height = image_size
    return {
        'exterior_image_1_left': _resize_image(base_rgb[..., ::-1] if base_rgb.shape[-1] == 3 else base_rgb, (width, height)),
        'wrist_image_left': _resize_image(wrist_rgb[..., ::-1] if wrist_rgb.shape[-1] == 3 else wrist_rgb, (width, height)),
        'joint_position': joint_positions[:7],
        'gripper_position': gripper_position,
        'actions': actions,
        'task': sample.get('task', task),
    }


def main(
    raw_dir: str,
    *,
    task: str,
    repo_id: str = REPO_ID,
    target_fps: int = 15,
    width: int = 320,
    height: int = 240,
    allow_missing_base: bool = False,
    swap_wrist_base_labels: bool = False,
    output_dir: str | None = None,
    push_to_hub: bool = False,
):
    raw_path = Path(raw_dir).expanduser()
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    output_path = Path(output_dir).expanduser() if output_dir is not None else HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type='panda',
        fps=target_fps,
        features={
            'exterior_image_1_left': {
                'dtype': 'image',
                'shape': (height, width, 3),
                'names': ['height', 'width', 'channel'],
            },
            'wrist_image_left': {
                'dtype': 'image',
                'shape': (height, width, 3),
                'names': ['height', 'width', 'channel'],
            },
            'joint_position': {
                'dtype': 'float32',
                'shape': (7,),
                'names': ['joint_position'],
            },
            'gripper_position': {
                'dtype': 'float32',
                'shape': (1,),
                'names': ['gripper_position'],
            },
            'actions': {
                'dtype': 'float32',
                'shape': (8,),
                'names': ['actions'],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    episode_dirs = [p for p in sorted(raw_path.iterdir()) if p.is_dir()]
    if not episode_dirs:
        raise ValueError(f'No episode directories found in {raw_path}')

    for episode_dir in episode_dirs:
        frames = _load_episode_frames(episode_dir)
        frames = _resample_by_time(frames, target_fps)
        if not frames:
            continue
        for sample in frames:
            dataset.add_frame(
                _extract_frame(
                    sample,
                    task=task,
                    image_size=(width, height),
                    allow_missing_base=allow_missing_base,
                    swap_wrist_base_labels=swap_wrist_base_labels,
                )
            )
        dataset.save_episode()

    if output_dir is not None:
        print(f'Dataset written to: {output_path}')

    if push_to_hub:
        dataset.push_to_hub(
            tags=['franka', 'teleop', 'panda'],
            private=False,
            push_videos=True,
            license='apache-2.0',
        )


if __name__ == '__main__':
    tyro.cli(main)
