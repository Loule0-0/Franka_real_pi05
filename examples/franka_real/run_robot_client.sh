#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLIENT_PY="${OPENPI_ROOT}/examples/franka_real/pi05_rtc_polymetis_client.py"

# =========================
# 固定配置: 直接修改这里
# =========================

# 语言指令。机器人会按照这个任务做推理。
PROMPT="pick up the fork"

# Policy server 地址。通常 policy server 和 robot client 都在本机。
POLICY_HOST="127.0.0.1"

# Policy server 端口。必须和 run_policy_server.sh 里的 SERVER_PORT 一致。
POLICY_PORT="8000"

# Polymetis robot service 地址。当前你的机器人服务跑在本机。
ROBOT_IP="172.16.0.100"

# Franka 机械臂服务端口。通常由 1_launch_robot.sh 启动。
FRANKA_PORT="50051"

# Franka 夹爪服务端口。通常由 2_launch_gripper.sh 启动。
GRIPPER_PORT="50052"

# 相机 ZMQ server 主机地址。当前 launch_camera_nodes.py 默认也是本机。
CAMERA_HOST="127.0.0.1"

# 自动探测相机的起始端口。launch_camera_nodes.py 默认从 5000 开始。
CAMERA_PORT_BASE="5000"

# 自动探测相机时，最多连续尝试多少个端口。
CAMERA_MAX_PORTS="4"

# 强制指定 exterior 相机端口。当前固定为 5000（第三视角）。
EXTERIOR_CAMERA_PORT="5000"

# 强制指定 wrist 相机端口。当前固定为 5001（腕部）。
WRIST_CAMERA_PORT="5001"

# 相机读图超时时间，单位毫秒。网络或相机不稳时可适当增大。
CAMERA_TIMEOUT_MS="2000"

# 运行时长，单位秒。设为 <=0 表示一直运行到 Ctrl+C。
DURATION_S="120"

# 机器人控制频率，单位 Hz。真机建议先从 15 开始，不要一开始追高。
CONTROL_HZ="15"

# Action queue 低于这个长度时就触发下一次推理。
QUEUE_REFILL_THRESHOLD="4"

# Action queue 的最大缓存长度。过大容易增加滞后。
MAX_QUEUE_SIZE="24"

# Realtime chunking 的重叠长度。更大更平滑，但响应会更慢。
EXECUTION_HORIZON="8"

# Realtime chunking 的融合强度。10.0 表示最强融合。
MAX_GUIDANCE_WEIGHT="10.0"

# Chunk 融合权重曲线。推荐保持 exp。
RTC_SCHEDULE="exp"

# 是否关闭 realtime chunking。0 表示开启，1 表示关闭。
DISABLE_RTC_STYLE="0"

# 初始推理延迟估计，单位控制步数。通常 3 比较合适。
INITIAL_INFERENCE_DELAY="3.0"

# 推理延迟 EMA 平滑系数。越大表示越平滑，越小表示越跟随当前耗时。
INFERENCE_DELAY_EMA_DECAY="0.7"

# 额外增加的延迟补偿步数。若明显抢拍或滞后，可微调这个值。
EXTRA_DELAY_STEPS="0.0"

# 每个控制周期允许的最大关节变化，单位弧度。越小越稳。
MAX_JOINT_STEP_RAD="0.03"

# 相邻两个周期之间，关节增量本身允许变化的最大幅度，单位弧度。
MAX_DELTA_CHANGE_RAD="0.015"

# 关节指令低通平滑系数。越小越平滑，越大越跟手。
JOINT_COMMAND_ALPHA="0.6"

# 队列空了之后是否保持当前位置/上一动作。1 推荐开启。
HOLD_POSITION_ON_EMPTY_QUEUE="1"

# 是否把 gripper 动作二值化。1 推荐开启，减少夹爪来回抖动。
BINARIZE_GRIPPER="1"

# gripper 二值化阈值。大于等于此值视为闭合。
GRIPPER_BINARY_THRESHOLD="0.5"

# gripper 命令死区。变化不足这个值就不下发新命令。
GRIPPER_DEADBAND="0.05"

# 两次 gripper 命令之间的最短时间间隔，单位秒。
GRIPPER_MIN_INTERVAL_S="0.2"

# gripper 运动速度。
GRIPPER_SPEED="1.0"

# gripper 力。
GRIPPER_FORCE="1.0"

# 是否在启动时先执行 robot.go_home()。1 会先回零，0 不回零。
GO_HOME="0"

# 是否在启动时先张开夹爪。1 推荐开启。
OPEN_GRIPPER_ON_START="1"

# =========================
# 基础检查
# =========================

echo "[INFO] OpenPI root: ${OPENPI_ROOT}"
echo "[INFO] Client script: ${CLIENT_PY}"

if [[ ! -f "${CLIENT_PY}" ]]; then
  echo "[ERROR] 找不到客户端脚本: ${CLIENT_PY}"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python，请先激活 franka_teleop 环境。"
  exit 1
fi

if ! python -c "import openpi_client, polymetis, zmq, torch" >/dev/null 2>&1; then
  echo "[ERROR] 当前环境缺少依赖（openpi_client/polymetis/zmq/torch 之一）。"
  echo "[HINT] 请在 franka_teleop 环境执行: pip install -e ${OPENPI_ROOT}/packages/openpi-client"
  exit 1
fi

check_tcp() {
  local host="$1"
  local port="$2"
  local name="$3"
  if ! python - "$host" "$port" "$name" <<'PY'
import socket, sys
host, port, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
s = socket.socket()
s.settimeout(1.5)
try:
    s.connect((host, port))
except Exception:
    print(f"[ERROR] 无法连接 {name}: {host}:{port}")
    sys.exit(1)
finally:
    s.close()
print(f"[INFO] 已连通 {name}: {host}:{port}")
PY
  then
    echo "[HINT] 请确认对应服务已启动。"
    exit 1
  fi
}

check_tcp "${POLICY_HOST}" "${POLICY_PORT}" "policy server"
check_tcp "${ROBOT_IP}" "${FRANKA_PORT}" "franka service"
check_tcp "${ROBOT_IP}" "${GRIPPER_PORT}" "gripper service"

# =========================
# 启动
# =========================

cd "${OPENPI_ROOT}"

CMD=(
  python "${CLIENT_PY}"
  --prompt "${PROMPT}"
  --policy_host "${POLICY_HOST}"
  --policy_port "${POLICY_PORT}"
  --robot_ip "${ROBOT_IP}"
  --franka_port "${FRANKA_PORT}"
  --gripper_port "${GRIPPER_PORT}"
  --camera_host "${CAMERA_HOST}"
  --camera_port_base "${CAMERA_PORT_BASE}"
  --camera_max_ports "${CAMERA_MAX_PORTS}"
  --camera_timeout_ms "${CAMERA_TIMEOUT_MS}"
  --duration_s "${DURATION_S}"
  --control_hz "${CONTROL_HZ}"
  --queue_refill_threshold "${QUEUE_REFILL_THRESHOLD}"
  --max_queue_size "${MAX_QUEUE_SIZE}"
  --execution_horizon "${EXECUTION_HORIZON}"
  --max_guidance_weight "${MAX_GUIDANCE_WEIGHT}"
  --rtc_schedule "${RTC_SCHEDULE}"
  --initial_inference_delay "${INITIAL_INFERENCE_DELAY}"
  --inference_delay_ema_decay "${INFERENCE_DELAY_EMA_DECAY}"
  --extra_delay_steps "${EXTRA_DELAY_STEPS}"
  --max_joint_step_rad "${MAX_JOINT_STEP_RAD}"
  --max_delta_change_rad "${MAX_DELTA_CHANGE_RAD}"
  --joint_command_alpha "${JOINT_COMMAND_ALPHA}"
  --gripper_binary_threshold "${GRIPPER_BINARY_THRESHOLD}"
  --gripper_deadband "${GRIPPER_DEADBAND}"
  --gripper_min_interval_s "${GRIPPER_MIN_INTERVAL_S}"
  --gripper_speed "${GRIPPER_SPEED}"
  --gripper_force "${GRIPPER_FORCE}"
)

if [[ "${EXTERIOR_CAMERA_PORT}" != "-1" ]]; then
  CMD+=(--exterior_camera_port "${EXTERIOR_CAMERA_PORT}")
fi
if [[ "${WRIST_CAMERA_PORT}" != "-1" ]]; then
  CMD+=(--wrist_camera_port "${WRIST_CAMERA_PORT}")
fi
if [[ "${DISABLE_RTC_STYLE}" == "1" ]]; then
  CMD+=(--disable_rtc_style)
fi
if [[ "${HOLD_POSITION_ON_EMPTY_QUEUE}" == "1" ]]; then
  CMD+=(--hold_position_on_empty_queue)
fi
if [[ "${BINARIZE_GRIPPER}" == "1" ]]; then
  CMD+=(--binarize_gripper)
fi
if [[ "${GO_HOME}" == "1" ]]; then
  CMD+=(--go_home)
fi
if [[ "${OPEN_GRIPPER_ON_START}" == "1" ]]; then
  CMD+=(--open_gripper_on_start)
fi

echo "[INFO] 即将启动机器人客户端"
echo "[INFO] 命令: ${CMD[*]}"
echo "[NOTE] 直接运行本脚本即可。如需调参，请修改脚本顶部配置块。"

exec "${CMD[@]}"
