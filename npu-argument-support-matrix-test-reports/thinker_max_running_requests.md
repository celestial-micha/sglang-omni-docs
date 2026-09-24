# 限制理解生成阶段的同时运行数量：thinker_max_running_requests

本次任务是检查：换用 NPU（昇腾计算设备）后，专门给 thinker 阶段设置的运行上限能否正常工作。

## 1. 验证对象与结论

Qwen3-Omni 把任务分成多个步骤。thinker 是负责理解输入、推理和生成文字等结果的阶段；语音生成还会使用其他阶段。

`thinker_max_running_requests` 是 `max_running_requests` 的 per-stage 写法——只给 thinker 设名额，不影响 talker_ar 等其他阶段。例如设为 3，就是让 thinker 最多同时计算 3 条请求，**不是让整个服务的所有阶段合起来只有 3 个名额**。可以给不同阶段设不同的值（如 thinker=32、talker_ar=16）。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `thinker_max_running_requests` | 需要，用来单独控制 thinker 的负载 | **[√] 已支持**。thinker 超上限正确排队（`#running-req: 3, #queue-req: 1`），per-stage 隔离确认（只设 thinker 不影响 talker_ar）。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8）×2 台 |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828） |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker） |
| 容器 | `codex-newmodels-base`（113.46.15.88）、`fth-qwen3-tts-async-v0519-cann91-sglang-151`（113.46.21.151） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设 `thinker.engine.max_running_requests=64` 和 `=0`，读回 overrides | 64 读回正确，0 被拒绝 | 通过：64 读回正确，0 被拒绝 | — |
| 端到端文本生成 | 设 `thinker.engine.max_running_requests=32`，启动后发文本请求 | HTTP 200 + 有内容 | 通过：HTTP 200，回 "Hello!"，prompt_tokens=15 | — |
| 端到端多模态生成 | 同上配置，发带 base64 PNG 图片的请求 | HTTP 200 + 有内容 + prompt_tokens 包含图片 token | 通过：HTTP 200，回 "A grayscale illustration of a pigeon..."，prompt_tokens=2563 | — |
| 超限排队 | 设 `thinker.engine.max_running_requests=3`，同时发 4 个长 prompt 请求（每条生成 62-69 tokens），看调度器日志 | running 不超过 3，多余的排队 | 通过：`Decode #running-req: 3, #queue-req: 1`——3 条在跑（卡在上限），1 条排队；某条结束后排队的进来，队列清空 | 4 条全 HTTP 200 |
| per-stage 隔离 | 只设 `thinker.engine.max_running_requests=3`，不设 talker_ar，看两个阶段是否各自独立 | thinker 被限为 3，talker_ar 用默认值不受影响 | 通过：thinker 卡在 3，talker_ar 用默认值 16 | talker_ar 限流也单独验通（详见 `max_running_requests.md`） |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（值传到 thinker 的 ServerArgs） | ✓ `test_cli.py::test_cli_thinker_max_running_requests_targets_thinker_in_both_variants` | ✓ |
| 取值校验（0 被拒，ge=1） | ✓ | ✓ |
| 实际限流行为（超上限排队） | ✗ UT 不启动 scheduler | ✓ `#running-req: 3, #queue-req: 1` |
| per-stage 隔离 | ✓ 路由层面验证 | ✓ 两个方向都验 |

UT 能验路由和取值校验，但不能验实际限流行为（需要 NPU 硬件 + 模型加载 + 并发请求）。实际限流通过端到端测试覆盖。
