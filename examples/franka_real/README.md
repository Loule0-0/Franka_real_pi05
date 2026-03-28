# Franka 真机部署

这个目录已经整理成“改脚本顶部配置，然后直接运行 `.sh`”的方式。

## 目录里的两个入口

- `run_policy_server.sh`
  - 在 `openpi` 环境运行。
  - 作用：启动 `pi05` 的 websocket policy server。
- `run_robot_client.sh`
  - 在 `franka_teleop` 环境运行。
  - 作用：读取机器人状态和相机图像，请求策略动作，并通过 Polymetis 控制 Franka。

## 运行前提

你现在这边已经有这些基础服务：

- 机器人服务：`/home/server/franka/franka_teleop/1_launch_robot.sh`
- 夹爪服务：`/home/server/franka/franka_teleop/2_launch_gripper.sh`
- 相机服务：`/home/server/franka/franka_teleop/teleop/experiments/launch_camera_nodes.py`
- 模型目录：`/home/server/Desktop/vla/model/pi05_base`

另外要确保：

- `openpi` 环境能导入 `openpi`
- `franka_teleop` 环境能导入 `polymetis`
- `franka_teleop` 环境已安装 `openpi-client`
  - 安装命令：`pip install -e /home/server/Desktop/vla/openpi/packages/openpi-client`

## 最简单运行方式

### 1. 在 openpi 环境启动策略服务

```bash
cd /home/server/Desktop/vla/openpi/examples/franka_real
./run_policy_server.sh
```

如果你想改模型路径、端口、默认 prompt，不要在命令行后面加参数，直接改 `run_policy_server.sh` 顶部配置块。

### 2. 在 franka_teleop 环境启动机器人客户端

```bash
cd /home/server/Desktop/vla/openpi/examples/franka_real
./run_robot_client.sh
```

如果你想改 prompt、控制频率、相机端口、平滑参数，也不要在命令行后面加参数，直接改 `run_robot_client.sh` 顶部配置块。

## 相机说明

机器人客户端现在支持自动探测相机：

- 如果探测到 1 个相机，就只连 1 个，并打印 `wrist=disabled`
- 如果探测到 2 个相机，就连 2 个，并打印实际端口映射
- 如果探测到超过 2 个，就只使用前 2 个，并打印被忽略的端口

默认探测参数写在 `run_robot_client.sh` 顶部：

- `CAMERA_PORT_BASE`
- `CAMERA_MAX_PORTS`
- `EXTERIOR_CAMERA_PORT`
- `WRIST_CAMERA_PORT`

规则是：

- 当 `EXTERIOR_CAMERA_PORT=-1` 且 `WRIST_CAMERA_PORT=-1` 时，程序自动探测
- 如果你手动填了某个端口，就优先按你手填的端口连接

## 推荐初始参数

`run_robot_client.sh` 里当前默认值已经偏保守，适合真机第一次跑：

- `CONTROL_HZ=15`
- `EXECUTION_HORIZON=8`
- `MAX_JOINT_STEP_RAD=0.03`
- `MAX_DELTA_CHANGE_RAD=0.015`
- `JOINT_COMMAND_ALPHA=0.6`
- `BINARIZE_GRIPPER=1`
- `HOLD_POSITION_ON_EMPTY_QUEUE=1`

如果后面你觉得动作太慢，可以再逐步调大；如果觉得不够稳，就先把步长和频率再往下收。

## 现在的 RTC 是什么

现在这套不是 LeRobot 模型内部的原生 RTC，而是兼容 OpenPI websocket `infer()` 接口的工程化 realtime chunking：

- 后台异步推理
- 根据真实推理耗时做延迟补偿
- 在 chunk 切换处做重叠融合
- 在执行层再做一步限幅和平滑

这样做的目的不是追求“和论文实现一模一样”，而是优先让你这套现有 OpenPI + Polymetis 真机链路跑得稳、顺、可调。

## 你现在应该怎么用

1. 先确认机器人、夹爪、相机服务都已经启动。
2. 在 `openpi` 环境执行 `./run_policy_server.sh`。
3. 在 `franka_teleop` 环境执行 `./run_robot_client.sh`。
4. 需要调参时，只改两个 `.sh` 文件顶部配置，不要在命令行追加额外参数。
