# 开启语音阶段的编译计算：talker_torch_compile

本次任务是检查：换用 NPU（昇腾计算设备）后，给语音生成阶段开启"编译计算"后，设置能否在 NPU 上正常工作。

## 1. 验证对象与结论

Qwen3-Omni 的 talker 负责生成声音所需的表示；它与负责理解、推理的 thinker 是不同阶段。

模型要反复做很多相似计算。"编译"可以理解为先把其中一些计算整理成便于执行的形式，再用这种形式运行。`talker_torch_compile` 控制是否尝试使用这条路径，目的是便于优化计算，但开启后不保证一定更快。

本参数只管 talker 这一阶段。 NPU 可以有自己的编译实现，因此理论上仍需要这个设置。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `talker_torch_compile` | 需要，NPU 也可能使用编译后的计算方式 | **需要开发支持**。根因同 `thinker_torch_compile`（不是 stage 专属问题）。 |

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
| 参数路由与取值校验 | 用 ConfigManager 设 `talker_ar.engine.enable_torch_compile=true`，读回 overrides | 值读回正确为 True | 通过：读回 True | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `talker_ar.engine.enable_torch_compile=true` + `disable_cuda_graph=true`，启动后发文本和图片请求 | HTTP 200 + 有内容 | 通过：两次 HTTP 200 | 但 torch.compile 从未被调用（静默空跑） |
| 编译是否真正发生 | 查 torchinductor 缓存 + 开 `TORCH_LOGS=+dynamo` 跑 | 有编译产物或 dynamo 输出 | 未通过：缓存为空，dynamo 零行输出 | 同 `thinker_torch_compile`：`disable_cuda_graph=true` 跳过了 graph capture → torch.compile 从未被调用 |
| 端到端启动（disable_cuda_graph=false） | 设 `enable_torch_compile=true` + `disable_cuda_graph=false` | torch.compile 被调用 | 未通过：torch.compile 被调用但 dynamo 崩溃 | 同 `thinker_torch_compile`：两层崩溃（`is_cuda()` + fake tensor 不认 NPU） |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由 | ✓ `test_pipeline.py::test_moss_tts_talker_torch_compile_dotted_flags_target_tts_engine` | ✓ |
| 取值校验 | ✓ | ✓ |
| torch.compile 是否被调用 | ✗ | ✓ 空跑 + 崩溃都验了 |
| dynamo 在 NPU 上能否追踪 | ✗ | ✓ 同 `thinker_torch_compile`（不是 stage 专属） |

UT 只验参数路由。torch.compile 的实际行为（空跑、dynamo 崩溃）需要端到端测试。根因同 `thinker_torch_compile`，不是 stage 专属问题——thinker 和 talker_ar 都走同一个 SGLang NPU 后端的 `patch_model_npu`，碰同一个 `fused_op.py` 的 `is_cuda()`，碰同一个 dynamo fake tensor 不认 NPU 设备的问题。
