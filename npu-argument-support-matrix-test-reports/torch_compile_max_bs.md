# 统一设置多个阶段的编译批大小：torch_compile_max_bs

本次任务是检查：换用 NPU（昇腾计算设备）后，各个适用的生成阶段的"编译批大小上限"设置能否在 NPU 上起作用。

## 1. 验证对象与结论

Qwen3-Omni 把任务分成多个阶段：thinker 负责理解、推理和生成文字等结果，talker 负责生成声音所需的表示。

这里的"编译"是先把部分模型计算整理成便于执行的形式，再用这种形式运行；本参数需要与编译开关（`torch_compile`）一起看。

模型一次计算可以同时处理多条请求，这些请求合起来叫一批，条数就是批大小。本参数限制与编译计算有关的批大小范围：batch_size ≤ `torch_compile_max_bs` 走编译图，> `torch_compile_max_bs` 走 eager。它用于把同一个上限设置给多个适用阶段。

**一条请求里有很长的文字，仍可能只算一条；字多不等于批大小大。** 这个设置也不是整个服务最多能接多少条请求。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `torch_compile_max_bs` | 需要，用于设置编译计算对应的批次范围 | **需要开发支持**。`torch_compile_max_bs` 依赖 `enable_torch_compile` 生效——而 `enable_torch_compile` 在 NPU 上当前不可用，`max_bs` 无从生效。 |

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
| 参数路由与取值校验 | 用 ConfigManager 设广播式 `torch_compile_max_bs=32` 和 `=0`，读回各 stage 的 overrides | 32 读回正确，0 被拒绝 | 通过：32 读回正确，0 被拒绝 | — |
| 端到端文本生成 | 设 `torch_compile_max_bs=8`（广播）+ `enable_torch_compile=true` + `disable_cuda_graph=true`，发文本和图片请求 | HTTP 200 + 有内容 | 通过：两次 HTTP 200 | 但 torch.compile 从未被调用（静默空跑），`max_bs` 无从生效 |
| 批大小边界 | 设 `torch_compile_max_bs=2`（广播）+ `enable_torch_compile=true` + `disable_cuda_graph=false`，发 4 个并发请求 | 超过 2 的走 eager | 未通过：torch.compile 在 NPU 上跑不通（dynamo 崩溃），`max_bs` 无从验证 | 根因同 `thinker_torch_compile` |
| 广播是否送到各 stage | 只设广播入口，查看各 stage 的最终值 | 每个目标 stage 得到正确上限 | 通过：函数级验证两个 stage 都收到正确值 | — |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（广播到各 stage） | ✓ `test_pipeline.py::test_s2pro_torch_compile_max_bs_rejects_non_positive` | ✓ |
| 取值校验 | ✓ | ✓ |
| 批大小实际生效 | ✗ 依赖 torch.compile 能跑 | ✗ torch.compile 跑不通 |
| 广播送达各 stage | ✓ 路由验证 | ✓ |

UT 只验路由和取值校验。批大小实际生效需要 torch.compile 先能跑通——而 torch.compile 在 NPU 上当前不可用（根因见 `thinker_torch_compile`）。torch.compile 修复后此项可补验。本参数的测试应与"服务同时运行多少条"的测试分开。
