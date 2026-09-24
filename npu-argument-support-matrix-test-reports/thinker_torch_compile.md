# 开启理解生成阶段的编译计算：thinker_torch_compile

本次任务是检查：换用 NPU（昇腾计算设备）后，给理解生成阶段开启"编译计算"后，设置能否在 NPU 上正常工作。

## 1. 验证对象与结论

Qwen3-Omni 的 thinker 负责理解输入、推理和生成文字等结果；语音生成还会使用其他阶段。

模型要反复做很多相似计算。"编译"可以理解为先把其中一些计算整理成便于执行的形式，再用这种形式运行。`thinker_torch_compile` 控制是否尝试使用这条路径，目的是便于优化计算，但开启后不保证一定更快。

本参数只管 thinker 这一阶段；它与语音阶段的编译开关要分别看。 NPU 可以有自己的编译实现（SGLang NPU 后端有 `patch_model_npu`，用 `npugraph_ex` 后端编译 forward），因此理论上仍需要这个设置。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `thinker_torch_compile` | 需要，NPU 也可能使用编译后的计算方式 | **需要开发支持**。函数级透传可用，但 torch.compile 在 NPU 上实际跑不通——根因有两层（详见测试用例）。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | Ascend910（64GB HBM ×8） |
| 软件 | CANN 9.1 + SGLang 0.5.18 + sglang_omni 0.1.4（c828）+ torch_npu 2.10.0 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（TP=2 thinker）、Fun-CosyVoice3-0.5B-2512 |
| 容器 | `codex-newmodels-base`（113.46.15.88） |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数路由与取值校验 | 用 ConfigManager 设 `thinker.engine.enable_torch_compile=true`，读回 overrides | 值读回正确为 True | 通过：读回 True | — |
| 端到端文本生成（disable_cuda_graph=true） | 设 `enable_torch_compile=true` + `disable_cuda_graph=true`，启动后发文本和图片请求 | HTTP 200 + 有内容 | 通过：两次 HTTP 200 | 但 torch.compile 从未被调用（详见下一行） |
| 编译是否真正发生 | 开 `TORCH_LOGS=+dynamo` 跑同上配置，查 torchinductor 缓存目录和 dynamo 日志输出 | 有编译产物或 dynamo 日志输出 | 未通过：torchinductor 缓存 delta=0（空）；`TORCH_LOGS=+dynamo` 零行输出。torch.compile 从未被调用 | 根因：torch.compile 只在 graph capture 内调用（`patch_model_npu` 上下文管理器），`disable_cuda_graph=true` 跳过了 graph capture → torch.compile 从未被调用。这是静默空跑——没有报错、没有警告，flag 设了但没用 |
| 端到端启动（disable_cuda_graph=false） | 设 `enable_torch_compile=true` + `disable_cuda_graph=false`，启动服务 | torch.compile 被调用，服务正常 | 未通过：torch.compile 被调用了（日志有 `torchair DeprecationWarning: npugraph_ex`），但 dynamo 追踪 forward 时崩溃 | 崩在两层，详见下两行 |
| dynamo 崩溃第一层：is_cuda() | dynamo 追踪到 `fused_op.py` 的 `is_cuda()` → `torch.cuda.is_available()`，这个函数被 `transfer_to_npu` 换成了 torch_npu.npu 的 Python 函数 | dynamo 能追踪 | 未通过：dynamo 对 `torch.cuda.is_available` 有专门追踪规则（按 C++ 内置函数处理），跟被换过的 Python 函数对不上 → 崩 | 修法：把 `is_npu()` 挪到 `is_cuda()` 前面（2 行）。`is_npu()` 调的是 `torch.npu.is_available()`（没被换过），dynamo 没有专门规则 → 直接执行 → 不崩。已验证此修法有效——第一颗雷拆掉后 dynamo 继续往前追踪 |
| dynamo 崩溃第二层：fake tensor | 拆掉第一颗雷后，dynamo 继续追踪，在把模型参数转成 fake tensor 时 | 能转换 NPU 设备上的 tensor | 未通过：报 `Unexpected type in sourceless builder`——dynamo 的 fake tensor 系统只认 `cuda` 和 `cpu`，不认 `npu` | 根因：PyTorch dynamo 有设备注册 API `register_interface_for_device()`，CUDA/XPU/CPU 都注册了，但 torch_npu 没注册 NPU。修法：torch_npu 团队实现 `NpuInterface` 并注册。需 torch_npu 团队修，SGLang 层改不了 |

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 参数路由（值传到 ServerArgs） | ✓ `test_cli.py::test_torch_compile_dotted_flags_reach_resolved_sglang_args` | ✓ |
| 取值校验 | ✓ | ✓ |
| torch.compile 是否被调用 | ✗ UT 不启动 SGLang | ✓ 查 torchinductor 缓存 + dynamo 日志（空跑 + 崩溃两层都验了） |
| dynamo 在 NPU 上能否追踪 | ✗ UT 不跑 dynamo | ✓ 两层崩溃都定位了 |
| `disable_cuda_graph` 对 torch.compile 的影响 | ✗ UT 不测跨参数交互 | ✓ 空跑（true）和崩溃（false）都验了 |

UT 只验参数路由，不验 torch.compile 实际行为。torch.compile 是否被调用、dynamo 是否崩溃需要端到端测试。另外 `enable_torch_compile=true` + `disable_cuda_graph=true` 的静默空跑问题没有 UT 拦截（建议加 validation 警告）。
