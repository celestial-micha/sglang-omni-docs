# 限制同时计算的请求数：max_running_requests

本次任务是检查：换用 NPU（昇腾计算设备）后，"最多同时处理几条请求"这个设置能否正常工作。

## 1. 验证对象与结论

一个服务可能同时收到很多任务，但不必把它们全部同时放进设备计算。`max_running_requests` 就像计算环节的名额数：设为 3，最多让 3 条请求同时运行，第 4 条来了就排队，等前面有任务结束空出名额再进来。

**同时运行 3 条，不等于一次只能提交 3 条，也不等于每秒只能完成 3 条。** NPU 的计算和内存资源也有限，仍然需要这个设置。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `max_running_requests` | 需要，用于控制同时进入计算的任务数量 | **[√] 已支持**。thinker 和 talker_ar 两个阶段的限流都验通了，超上限正确排队，不报错不拒绝。per-stage 隔离也确认（只设一个阶段不影响另一个）。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8）×2 台 |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828） |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker）+ Fun-CosyVoice3-0.5B-2512 |
| 容器 | `codex-newmodels-base`（113.46.15.88）、`fth-qwen3-tts-async-v0519-cann91-sglang-151`（113.46.21.151） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设 `max_running_requests=32` 和 `=0`，读回 `engine.overrides()` | 32 读回正确，0 被拒绝 | 通过：32 读回正确，0 被拒绝 | — |
| 端到端文本生成 | 设 `max_running_requests=32`，Qwen3-Omni-30B 启动后发文本请求 "Say hello in one short sentence." | HTTP 200 + 有内容 | 通过：HTTP 200，回 "Hello!"，prompt_tokens=15 | — |
| 端到端多模态生成 | 同上配置，发带 base64 PNG 图片的请求 | HTTP 200 + 有内容 + prompt_tokens 包含图片 token | 通过：HTTP 200，回 "A grayscale illustration of a pigeon..."，prompt_tokens=2563 | 图片经 image_encoder 编码后入 thinker KV |
| thinker 超限排队 | 设 `max_running_requests=3`，同时发 4 个长 prompt 请求（每条生成 62-69 tokens），看调度器日志 | running 不超过 3，多余的排队，空位后继续 | 通过：`Decode #running-req: 3, #queue-req: 1`——3 条在跑（卡在上限），1 条排队；某条结束后排队的进来，队列清空 | 4 条全 HTTP 200 |
| talker_ar 超限排队 | 只设 `talker_ar.engine.max_running_requests=2`（thinker 不设），发 4 个并发音频请求（`max_tokens=256` 确保生成时间长），看调度器日志 | running 不超过 2，多余的排队 | 通过：`Decode #running-req: 2, #queue-req: 2` 持续 2 分钟——2 条在跑（卡在上限），2 条排队；某条结束后队列从 2→1→0 | 音频请求（`modalities:["text","audio"]`）确保请求经过 talker_ar 阶段 |
| per-stage 隔离（只设 thinker） | 只设 `thinker.engine.max_running_requests=3`，不设 talker_ar，看两个阶段是否各自独立 | thinker 被限为 3，talker_ar 用默认值不受影响 | 通过：两个阶段都正常启动；thinker 卡在 3，talker_ar 启动日志无 `max_running_requests`（用默认值 16） | — |
| per-stage 隔离（只设 talker_ar） | 只设 `talker_ar.engine.max_running_requests=2`，不设 thinker，看两个阶段是否各自独立 | talker_ar 被限为 2，thinker 用默认值不受影响 | 通过：talker_ar 卡在 2（`#running-req: 2`），thinker 同期 running=4（默认 16，不限） | — |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（值从 YAML 传到 ServerArgs） | ✓ `test_cli.py::test_cli_thinker_max_running_requests_targets_thinker_in_both_variants` | ✓ |
| 取值校验（0 被拒，ge=1） | ✓ | ✓ |
| 实际限流行为（超上限排队） | ✗ UT 不启动 scheduler | ✓ thinker（`#running-req: 3`）+ talker_ar（`#running-req: 2`） |
| per-stage 隔离 | ✓ 路由层面验证 | ✓ 两个方向都验 |

UT 能验路由和取值校验，但不能验实际限流行为（需要 NPU 硬件 + 模型加载 + 并发请求 + 调度器日志）。实际限流通过端到端测试覆盖。
