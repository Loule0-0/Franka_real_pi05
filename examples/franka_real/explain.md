# OpenPI Franka Real 流程说明

这份文档解释当前 `examples/franka_real` 这条真实机器人链路的工作流程，重点覆盖：

- 你的三台设备分别扮演什么角色
- `pi05` policy server 的输入输出格式
- 观测、状态、动作在 server 端经过了哪些处理
- `instruction/prompt` 是如何传入模型的
- client 端是如何通过 Polymetis 控制 Franka 的
- RTC（Real-Time Chunking）的原理和在本项目里的具体实现

本文基于当前仓库里的实际代码，而不是泛化描述。

## 1. 当前系统里的三类 IP

你当前这套系统里至少有三个不同层级的地址：

- `172.16.0.2`
  真实 Franka 机械臂本体的 FCI 地址。这个地址是 Polymetis server 进程在服务端配置里使用的，不是 OpenPI client 直接连接的目标。
- `172.16.0.100`
  跑 Polymetis robot/gripper gRPC 服务的控制机地址。`RobotInterface` / `GripperInterface` 实际连的是它。
- `172.16.0.200`
  你现在跑 OpenPI robot client 的本机地址。

关键点：

- OpenPI client 不直接连 `172.16.0.2`
- OpenPI client 连的是 Polymetis server，也就是 `172.16.0.100:50051`
- 夹爪服务也是连 `172.16.0.100`，当前你的环境默认端口是 `50052`

对应证据：

- Polymetis 示例里 `RobotInterface(ip_address="172.16.0.100")`，见 `polymetis/examples/0_set_joint_positions.py`
- Polymetis robot server 硬件配置里机器人本体 IP 是 `172.16.0.2`，见 `polymetis/conf/robot_client/franka_hardware.yaml`
- Polymetis gripper 配置里机器人本体 IP 也是 `172.16.0.2`，见 `polymetis/conf/gripper/franka_hand.yaml`

## 2. 两个进程分别做什么

这条链路主要有两个常驻进程：

1. `run_policy_server.sh`
   在 GPU 机器上启动 OpenPI websocket policy server
2. `run_robot_client.sh`
   在机器人侧启动实时 client，采集相机和关节状态，向 policy server 请求动作，再把动作发给 Polymetis

### 2.1 policy server

启动命令来自 `run_policy_server.sh`：

```bash
python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_droid \
  --policy.dir /home/server/Desktop/vla/model/pi05_base
```

它做的事情是：

- 根据 `pi05_droid` 配置构造 policy
- 从 checkpoint 目录加载权重
- 从 checkpoint 下的 `assets/droid` 加载归一化统计量
- 启动 websocket server，监听 `0.0.0.0:8000`

代码入口见：

- `scripts/serve_policy.py`
- `src/openpi/policies/policy_config.py`
- `src/openpi/serving/websocket_policy_server.py`

### 2.2 robot client

`run_robot_client.sh` 做三件事：

1. 检查 policy server 是否连通
2. 检查 Polymetis robot/gripper 服务是否连通
3. 启动 `pi05_rtc_polymetis_client.py`

你当前日志说明这三条链路已经打通：

- policy server: `127.0.0.1:8000`
- franka service: `172.16.0.100:50051`
- gripper service: `172.16.0.100:50052`

## 3. 你刚才那次运行是否正常

从你给的两段日志看，整体是按计划执行的。

### 3.1 policy server 日志

这几行是正常的：

- `INFO:websockets.server:connection open`
- `Connection from ('127.0.0.1', 59118) opened`

说明：

- robot client 已经成功连上 websocket policy server
- 连接是本机回环地址，因为你的 policy server 就跑在本机

这几行是首次编译/选择 cuDNN 算法时的慢启动提示：

- `Trying algorithm ... is taking a while...`
- `The operation took 5.666s`

这通常表示：

- JAX/XLA 第一次跑到某个卷积时在做 kernel 选择或编译
- 它不是逻辑错误
- 它主要影响 warmup 或首批推理，不表示模型输出异常

### 3.2 robot client 日志

以下都说明运行状态正常：

- 成功探测到外部相机 `5000`
- 执行了 `start_joint_impedance()`
- 完成了一次 warmup infer
- 主循环正常运行
- 推理延迟稳定在约 `63-65ms`
- 队列长度在波动但没有掉空

其中：

- `delay_steps=1` 或 `2` 说明系统在根据实际推理耗时做延迟补偿
- `drop=1/2` 表示丢弃了已经“过时”的 action chunk 前缀
- `overlap=3` 表示 RTC 对新旧 chunk 做了 3 步重叠融合

这些值说明 RTC 正在工作，而且工作方式符合预期。

### 3.3 这两个 warning 怎么看

这两条一般不是致命问题：

- `Warning: Using default version.`
- `Failed to load 'libtorchscript_pinocchio.so' from CONDA_PREFIX, loading from default build directory instead`

含义是：

- polymetis 的某些动态库没有从当前 `CONDA_PREFIX` 找到
- 它退回到了源码 build 目录里的库

只要后续控制正常、没有崩溃、关节状态能读、命令能发，这种 fallback 通常可以接受。

## 4. websocket 协议层在传什么

OpenPI client 和 policy server 之间用的是 websocket + msgpack，不是 HTTP RPC，也不是 gRPC。

### 4.1 序列化方式

使用的是 `msgpack`，并对 `numpy.ndarray` 做了自定义打包：

- 数组被序列化成 `data + dtype + shape`
- 不允许 object array / complex / void dtype

实现见 `packages/openpi-client/src/openpi_client/msgpack_numpy.py`。

### 4.2 连接建立后的第一帧

client 连接后，server 会先发一帧 metadata：

- 代码在 `WebsocketPolicyServer._handler`
- client 在 `_wait_for_server()` 里先 `recv()` 这一帧

也就是说，第一次收到的不是 action，而是 policy metadata。

### 4.3 每次推理请求的请求体

`WebsocketClientPolicy.infer(obs)` 的行为非常直接：

1. 把 `obs` 用 msgpack 打包
2. 通过 websocket 发送
3. 等 server 返回二进制响应
4. 解包成 Python dict

实现见 `packages/openpi-client/src/openpi_client/websocket_client_policy.py`。

## 5. 当前 pi05 server 期待什么输入格式

你当前 server 配置是 `pi05_droid`，因此它期待的是 DROID 风格输入。

最外层输入 dict 需要这些键：

```python
{
  "observation/exterior_image_1_left": uint8[h, w, 3],
  "observation/wrist_image_left": uint8[h, w, 3],
  "observation/joint_position": float32[7],
  "observation/gripper_position": float32[1],
  "prompt": str,
}
```

这是由 `src/openpi/policies/droid_policy.py` 的 `DroidInputs` 定义的。

你当前 client 在 `_observe()` 里构造的就是这个结构：

- `observation/exterior_image_1_left`
- `observation/wrist_image_left`
- `observation/joint_position`
- `observation/gripper_position`
- `prompt`

对应代码在 `examples/franka_real/pi05_rtc_polymetis_client.py`。

## 6. client 侧观测是怎么构造的

### 6.1 图像

client 从 ZMQ camera server 读取 BGR 图像，然后：

1. BGR 转 RGB
2. resize + pad 到 `224 x 224`

因此 policy server 看到的图像已经是：

- RGB
- `224 x 224`
- 3 通道

这一步发生在 client 本地，不是在 policy server 上做。

### 6.2 关节状态

client 用：

```python
self.robot.get_joint_positions()
```

从 Polymetis 读取 7 维关节位置，作为：

```python
"observation/joint_position": q
```

### 6.3 夹爪状态

client 用：

```python
self.gripper.get_state().width
```

读取夹爪开口宽度，然后转成一个“闭合度”：

```python
gripper_closed = 1.0 - clip(width / MAX_OPEN_M, 0, 1)
```

所以：

- 张开时接近 `0`
- 闭合时接近 `1`

随后作为：

```python
"observation/gripper_position": np.array([gripper_closed], dtype=np.float32)
```

## 7. server 端收到输入后，经历了哪些处理

当前 `create_trained_policy()` 组装出来的输入变换顺序是：

1. `InjectDefaultPrompt(default_prompt)`
2. `data_config.data_transforms.inputs`
3. `Normalize(norm_stats, use_quantiles=...)`
4. `data_config.model_transforms.inputs`

见 `src/openpi/policies/policy_config.py`。

对于你当前的 `pi05_droid`，可具体展开为：

1. 注入默认 prompt
2. `DroidInputs`
3. 对 state 做 quantile normalization
4. `InjectDefaultPrompt`（模型级，通常是 no-op，因为 prompt 已存在）
5. `ResizeImages(224, 224)`
6. `TokenizePrompt(..., discrete_state_input=True)`
7. `PadStatesAndActions(model_action_dim=32)`

### 7.1 `DroidInputs` 做了什么

`DroidInputs` 会把原始 DROID 风格输入重组成模型内部格式：

- `state = concat(joint_position[7], gripper_position[1])`
- 把图像组织成 `image` 字典
- 为每个相机生成 `image_mask`

对于 `pi05`，它映射为：

- `base_0_rgb`
- `left_wrist_0_rgb`
- `right_wrist_0_rgb`

其中第三路右腕图像在你的场景里没有，直接补零图，并把 mask 置为 `False`。

因此模型真正吃到的状态是 8 维：

- 前 7 维是关节位置
- 第 8 维是 gripper 闭合度

### 7.2 归一化

你当前的 `pi05_droid` 使用 quantile normalization，而不是普通 z-score。

公式是：

```text
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
```

也就是把状态映射到大致 `[-1, 1]` 区间。

这里的统计量来自 checkpoint 自带的 `assets/droid`，不是你 client 现场计算的。

重要点：

- 主要被归一化的是 `state`
- 图像不经过这个 `Normalize` 数值标准化逻辑
- `actions` 在训练时也会按同一套统计量处理；推理时是输出后再反归一化

### 7.3 prompt 和 state 如何进入 pi05

`pi05` 和 `pi0` 的一个关键差异是：

- `pi0` 可以把 state 作为连续输入
- `pi05` 默认把 state 离散化后塞进文本 token 序列

`TokenizePrompt(..., discrete_state_input=True)` 会做：

1. 清洗 prompt 文本
2. 将已经归一化后的 state 按 256 个 bin 离散化
3. 拼成字符串：

```text
Task: pick up the fork, State: 132 118 ...;
Action:
```

4. 再送入 PaliGemma tokenizer

也就是说，在 `pi05` 里：

- instruction 不是单独一条 side input
- 它和离散化 state 一起组成文本前缀

这也是为什么归一化发生在 tokenization 之前：state 需要先落到接近 `[-1, 1]`，再离散成 256 档。

### 7.4 图像处理

`ResizeImages(224, 224)` 会对 `image` 字典里的每路图像执行 `resize_with_pad`。

你 client 本地已经做过一次 `224x224` 预处理，所以这一步通常只是再次保证尺寸一致，不会改变接口含义。

### 7.5 pad 到模型 action 维度

你的真实状态和最终动作只用前 8 维，但 `pi05` 基础模型的 `action_dim` 是 32。

所以 `PadStatesAndActions(32)` 会把：

- `state[8]` pad 成 `state[32]`
- 推理输出的动作内部也是 `32` 维

后 24 维只是为了匹配基础模型接口，不是你的 Franka 真正执行的自由度。

## 8. pi05 server 的输出格式

`Policy.infer()` 产生的原始返回结构是：

```python
{
  "state": ...,
  "actions": ...,
}
```

其中：

- `state` 是变换后的内部 state
- `actions` 是模型输出的 action chunk

随后会经过 output transforms。

对于 `pi05_droid`，输出变换顺序是：

1. `model_transforms.outputs`
2. `Unnormalize(norm_stats, use_quantiles=True)`
3. `data_transforms.outputs`

展开以后主要是：

1. 对动作做反归一化
2. `DroidOutputs` 只保留前 8 维

最终 websocket 返回给 client 的核心是：

```python
{
  "actions": float32[action_horizon, 8],
  "policy_timing": {...},
  "server_timing": {...},
}
```

当前 `pi05_droid` 的 `action_horizon=15`，见 `src/openpi/training/config.py` 的 `pi05_droid` 配置。

因此每次 server 返回的是：

- `15 x 8` 的动作块

8 维含义是：

- 前 7 维：关节相关动作
- 第 8 维：夹爪动作

## 9. action 在 server 端是如何反归一化的

`Unnormalize` 对 quantile norm 的逆变换是：

```text
x = (x_norm + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

这一步非常关键，因为模型内部输出是归一化空间的动作，而 client 最终需要的是物理量空间里的动作。

对你这条链路来说，重要结果是：

- 返回到 client 的 `actions[:, :8]` 已经是物理语义上的关节/夹爪量
- 不是仍处于 `[-1, 1]` 归一化空间的值

## 10. client 如何调用 Polymetis

你的 client 并没有自己实现 gRPC proto，而是直接使用 Polymetis Python SDK：

```python
self.robot = RobotInterface(ip_address=args.robot_ip, port=args.franka_port)
self.gripper = GripperInterface(ip_address=args.robot_ip, port=args.gripper_port)
```

因此：

- `RobotInterface` 连接 Polymetis arm server
- `GripperInterface` 连接 Polymetis gripper server

### 10.1 初始化阶段

启动后 client 会做：

1. 可选 `go_home()`
2. `start_joint_impedance()`
3. 可选先张开夹爪
4. 做一次 warmup inference

其中真正进入实时控制模式的是：

```python
self.robot.start_joint_impedance()
```

### 10.2 每步执行动作

主循环每一步拿到一个 8 维 action 后：

1. 调 `_compute_joint_command()`
2. 计算当前关节到目标关节的受限增量
3. 调用：

```python
self.robot.update_desired_joint_positions(torch.from_numpy(q_cmd).float())
```

这说明你的 arm 控制模式是：

- 不是每步都 `move_to_joint_positions`
- 而是在 joint impedance controller 下不断更新期望关节位置

对于夹爪则是：

```python
self.gripper.goto(width=..., speed=..., force=...)
```

### 10.3 为什么这和直接连机器人本体不同

因为 OpenPI client 只是 Polymetis 的上层调用者：

- OpenPI client -> Polymetis gRPC server
- Polymetis server -> libfranka / Franka FCI -> 真实机器人

所以 OpenPI client 不需要知道 `172.16.0.2` 的细节，它只需知道 `172.16.0.100:50051/50052`。

## 11. client 端自己的动作安全措施

除了 Polymetis server 自带的安全层，你的 OpenPI client 自己又加了一层比较保守的限幅。

### 11.1 基础裁剪

在 `_sanitize_actions()` 里：

- 前 7 维关节动作先裁到 `JOINT_LIMIT_LOW/HIGH`
- gripper 动作裁到 `[0, 1]`

### 11.2 chunk 内相邻步限幅

对 action chunk 内部的每相邻两步：

```python
delta = clip(next - prev, -max_joint_step_rad, max_joint_step_rad)
```

默认是：

- `MAX_JOINT_STEP_RAD = 0.03`

### 11.3 当前状态到目标状态的单步限幅

在 `_compute_joint_command()` 里：

- 先算 `raw_delta = q_target - q_cur`
- 再裁成每步最多 `0.03 rad`

### 11.4 “加速度”限幅

不是只限制位置差，还限制“增量变化幅度”：

```python
delta = clip(delta, last_joint_delta - accel_limit, last_joint_delta + accel_limit)
```

默认：

- `MAX_DELTA_CHANGE_RAD = 0.015`

这会让指令变化更平滑，减少突然抽动。

### 11.5 低通滤波

若有上一帧命令，则：

```python
q_cmd = alpha * q_cmd + (1 - alpha) * last_sent_joint_cmd
```

默认：

- `JOINT_COMMAND_ALPHA = 0.6`

### 11.6 队列空时保持动作

如果推理暂时没跟上，client 会回退到：

- 上一个已执行 action
- 或者当前实际关节位置

不会直接发空命令。

### 11.7 夹爪防抖

夹爪这边还有：

- 二值化
- deadband
- 最小命令间隔

用来减少夹爪来回抖动。

## 12. Polymetis server 自带的更底层安全层

你这套 Polymetis 硬件配置里其实还有更强的 server 侧保护，定义在：

- `polymetis/conf/robot_client/franka_hardware.yaml`

它至少包含：

- `limit_rate: true`
- 低通截止频率 `lpf_cutoff_frequency: 100`
- 笛卡尔工作空间上下界
- 关节位置限制
- 关节速度限制
- 力矩限制
- collision behavior 阈值
- 激活的 `safety_controller`

也就是说，你现在不是裸奔：

- OpenPI client 有一层软限幅
- Polymetis server 又有一层硬件侧安全限制

## 13. RTC 是什么

RTC 指这里的 `Realtime Chunking`。

背景是：

- 模型每次推理不会只输出一步动作
- 而是一次输出一整段 action chunk
- 但真实系统里推理有延迟，新的 chunk 回来时，旧 chunk 往往已经执行了一部分

如果粗暴替换成新 chunk，会有两个问题：

- 新 chunk 前几步可能已经“过时”
- 新旧 chunk 拼接处可能不连续，导致动作突变

RTC 就是为了解决这两个问题。

## 14. 当前实现里的 RTC 是怎么工作的

当前实现是 `RealtimeActionQueue` + `RTCConfig`。

### 14.1 第一步：估计推理延迟

在 `_infer_loop()` 中，每次推理都会记录耗时：

```python
measured_delay = infer_s / control_dt
```

如果：

- 控制频率 `15 Hz`
- 每步周期约 `66.7 ms`
- 推理耗时 `64 ms`

那么推理延迟大约就是 `1` 步。

系统再对它做 EMA 平滑：

```python
estimated_inference_delay = decay * old + (1 - decay) * measured_delay
```

然后四舍五入成整数步数。

### 14.2 第二步：丢弃已经过时的 chunk 前缀

在 `merge_chunk()` 中：

```python
stale_steps = min(inference_delay, len(chunk))
fresh_chunk = chunk[stale_steps:]
```

例如推理延迟是 2 步，那么新 chunk 的前 2 步在“到达时”理论上已经错过最佳执行时机，于是直接丢掉。

这就是你日志里 `drop=1` 或 `drop=2` 的来源。

### 14.3 第三步：与旧队列前缀做重叠融合

如果队列里还有旧动作，RTC 不会直接把新 chunk 硬插进去，而是对前若干步做 blending：

```python
fresh_chunk[idx] = w * current[idx] + (1 - w) * fresh_chunk[idx]
```

其中：

- `current[idx]` 是旧队列中尚未执行的动作
- `fresh_chunk[idx]` 是新推理出的动作
- `w` 是一个随步数衰减的融合权重

当前你配置的是：

- `execution_horizon = 8`
- `max_guidance_weight = 10.0`
- `rtc_schedule = exp`

也就是：

- 在重叠前缀上采用指数衰减权重
- 越靠前越偏向旧动作
- 越往后越偏向新动作

这样拼接处就不会突然跳变。

### 14.4 第四步：更新队列

融合后的前缀覆盖旧队列前缀，其余新动作接在后面，形成新的待执行队列。

所以 RTC 的本质是：

- 根据推理延迟删掉过时动作
- 在新旧 chunk 交接处做平滑融合

## 15. 你日志里的 RTC 数值怎么理解

比如这行：

```text
[run] step=15 queue=9 infer=64.2ms delay_steps=2 drop=2 overlap=3
```

含义是：

- 这次最近一次推理耗时 `64.2ms`
- 估计相当于 `2` 个控制步的延迟
- 因此新 chunk 前 2 步被丢掉
- 新旧队列有 3 步被拿来做重叠融合

这正是 RTC 应有的行为。

## 16. 为什么 warmup 后 infer 只有 64ms，而 server 一开始打印了 5.6s

因为这两个时间不是同一件事：

- server 里 `5.6s` 是首次图编译/算子选择的慢启动
- client 日志里的 `64ms` 是 steady-state 下每次正常推理的耗时

通常流程是：

1. 首次 warmup 很慢
2. 编译完成后进入稳定推理
3. 后续每次 infer 基本维持几十毫秒

你当前日志正符合这个模式。

## 17. 当前链路的一个重要维度对齐

你真实机器人实际执行的是 8 维：

- 7 个关节
- 1 个 gripper

但模型内部用的是 32 维动作空间。

这不是 bug，而是基础模型接口设计：

- 输入 state 会 pad 到 32 维
- 模型输出动作 chunk 也是 32 维
- 最后再通过 `DroidOutputs` 截成前 8 维给你当前机器人执行

因此你可以把这理解为：

- `pi05 base` 提供通用 32 维 latent action 接口
- `DROID adapter` 只取和当前机器人相关的前 8 维

## 18. 这条链路的端到端摘要

每个控制周期，本质上是下面这条路径：

1. 本机 client 从相机读图，从 Polymetis 读关节和夹爪状态
2. client 把它们整理成 DROID 风格 observation，并附上语言 instruction
3. observation 通过 websocket + msgpack 发给本机 `pi05` policy server
4. server 用 `DroidInputs` 把输入改造成模型格式
5. server 用 checkpoint 里的 `droid` 统计量对 state 做 quantile normalization
6. server 把 prompt 和离散化后的 state 一起 tokenizer 成 `pi05` 的文本前缀
7. server 运行模型，输出 15 步的 action chunk
8. server 对动作反归一化，并只保留前 8 维返回给 client
9. client 根据推理延迟做 RTC：丢 stale prefix，和旧队列前缀融合
10. client 逐步取出 action，经过自身限幅和平滑后调用 Polymetis
11. Polymetis 再通过底层安全控制和 FCI 驱动真实机器人

## 19. 你当前配置下最值得记住的数字

- policy server: `127.0.0.1:8000`
- Polymetis robot server: `172.16.0.100:50051`
- Polymetis gripper server: `172.16.0.100:50052`
- 真实机器人本体: `172.16.0.2`
- 控制频率: `15 Hz`
- 当前稳态推理延迟: 约 `64 ms`
- `pi05_droid` action horizon: `15`
- client 实际执行 action dim: `8`
- model 内部 action dim: `32`

## 20. 当前日志对应的结论

你这次运行结果说明：

- policy server 正常启动并接受连接
- OpenPI client 已成功连上 Polymetis robot 和 gripper 服务
- 相机链路正常
- warmup 成功
- steady-state 推理正常
- RTC 正常工作
- 目前没有看到“没按计划执行”的迹象

如果你后面还想继续深入，下一份最值得补的文档是：

- 每个关键配置项怎么调
- 调大/调小后会引起什么行为变化
- 在真机上如何更保守地限制动作
