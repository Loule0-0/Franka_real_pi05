#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# =========================
# 固定配置: 直接修改这里
# =========================

# OpenPI 模型目录。你的 pi05 模型当前放在这里。
MODEL_DIR="/home/server/Desktop/vla/model/pi05_base"

# OpenPI 策略配置名。pi05 基础模型保持这个即可。
POLICY_CONFIG="pi05_droid"

# Websocket policy server 监听端口。机器人端脚本要和这里一致。
SERVER_PORT="8000"

# 可选默认语言指令。留空表示机器人端传什么 prompt 就用什么。
DEFAULT_PROMPT=""

# =========================
# 基础检查
# =========================

echo "========== OpenPI Policy Server =========="
echo "Root:   ${OPENPI_ROOT}"
echo "Model:  ${MODEL_DIR}"
echo "Config: ${POLICY_CONFIG}"
echo "Port:   ${SERVER_PORT}"
echo "=========================================="

[[ -d "${OPENPI_ROOT}" ]] || { echo "[ERROR] openpi 根目录不存在: ${OPENPI_ROOT}"; exit 1; }
[[ -f "${OPENPI_ROOT}/scripts/serve_policy.py" ]] || { echo "[ERROR] 缺少脚本: scripts/serve_policy.py"; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "[ERROR] 模型目录不存在: ${MODEL_DIR}"; exit 1; }

if [[ ! -d "${MODEL_DIR}/params" && ! -f "${MODEL_DIR}/model.safetensors" ]]; then
  echo "[ERROR] 模型目录结构不合法: 既没有 params/ 也没有 model.safetensors"
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[ERROR] 未找到 python，请先激活 openpi 环境。"
  exit 1
fi

if ! python -c "import openpi, tyro" >/dev/null 2>&1; then
  echo "[ERROR] 当前环境无法导入 openpi/tyro，请先激活 openpi 环境。"
  exit 1
fi

if ss -ltn | awk '{print $4}' | grep -q ":${SERVER_PORT}$"; then
  echo "[ERROR] 端口 ${SERVER_PORT} 已被占用。"
  exit 1
fi

# =========================
# 启动
# =========================

cd "${OPENPI_ROOT}"

CMD=(
  python scripts/serve_policy.py
  --port "${SERVER_PORT}"
  policy:checkpoint
  --policy.config "${POLICY_CONFIG}"
  --policy.dir "${MODEL_DIR}"
)

if [[ -n "${DEFAULT_PROMPT}" ]]; then
  CMD+=(--default_prompt "${DEFAULT_PROMPT}")
fi

echo "[INFO] 即将启动 policy server"
echo "[INFO] 命令: ${CMD[*]}"
echo "[NOTE] 直接运行本脚本即可。如需调参，请修改脚本顶部配置块。"

exec "${CMD[@]}"
