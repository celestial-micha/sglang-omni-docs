# 一次设置多个阶段的编译开关：torch_compile

本次任务是检查：换用 NPU（昇腾计算设备）后，给各个适用的生成阶段开启"编译计算"后，设置能否在 NPU 上正常工作。

## 1. 验证对象与结论

Qwen3-Omni 把任务分成多个阶段：thinker 负责理解、推理和生成文字等结果，talker 负责生成声音所需的表示。

模型要反复做很多相似计算。"编译"可以理解为先把其中一些计算整理成便于执行的形式，再用这种形式运行。`torch_compile` 控制是否尝试使用这条路径，目的是便于优化计算，但开启后不保证一定更快。

这个参数把同类开关统一发给多个适用的生成阶段（thinker + talker_ar），避免逐个设置。它不意味着服务里的所有组件都被编译。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `torch_compile` | 需要，NPU 也可能使用编译后的计算方式 | **需要开发支持**。根因同 `thinker_torch_compile`（广播写法落到各 stage 后行为一致）。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8） |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828）+ torch_npu 2.10.0 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker + talker_ar） |
| 容器 | `codex-newmodels-base`（113.46.15.88） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设广播式 `enable_torch_compile=true`，读回 thinker 和 talker_ar 的 overrides | 两个 stage 都收到 True | 通过：两个 stage 都读到 True | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `torch_compile=true`（广播）+ `disable_cuda_graph=true`，启动后发文本和图片请求 | HTTP 200 + 有内容 | 通过：两次 HTTP 200 | 但 torch.compile 从未被调用（静默空跑） |
| 编译是否真正发生 | 查 torchinductor 缓存 + 开 `TORCH_LOGS=+dynamo` 跑 | 有编译产物或 dynamo 输出 | 未通过：缓存为空，dynamo 零行输出 | 同 `thinker_torch_compile` |
| 端到端启动（disable_cuda_graph=false） | 设 `torch_compile=true`（广播）+ `disable_cuda_graph=false` | torch.compile 被调用 | 未通过：dynamo 崩溃 | 同 `thinker_torch_compile`：两层崩溃（`is_cuda()` + fake tensor） |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（广播到各 stage） | ✓ `test_cli.py::test_torch_compile_dotted_flags_reach_resolved_sglang_args` | ✓ |
| 取值校验 | ✓ | ✓ |
| torch.compile 是否被调用 | ✗ | ✓ 空跑 + 崩溃都验了 |
| 广播是否送到各 stage | ✓ 路由验证 | ✓ |

UT 只验参数路由。torch.compile 实际行为需要端到端测试。根因同 `thinker_torch_compile`（广播写法落到各 stage 后行为一致，不是 stage 专属问题）。
