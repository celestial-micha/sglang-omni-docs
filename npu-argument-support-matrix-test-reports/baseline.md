# NPU 参数支持矩阵：初始验证记录

- 记录日期：2026-09-10
- 来源：[Issue #1597](https://github.com/sgl-project/sglang-omni/issues/1597)
- 汇总更新日期：2026-09-30
- 本文性质：上游矩阵快照与同目录已完成的参数实测报告汇总；本次更新未重新执行 NPU 测试。
- 已收录报告：23 项，其中 16 项已支持（含报告注明的验证范围与限制）、6 项需要开发支持、1 项已删除而无需支持。

## 上游矩阵与参数支持验证

以下保留 Issue 原始列、默认值及标记。原 `supported` 列中的 `[√]`、`[x]` 和空白均为来源标记，不额外推断 UT 或端到端测试结果。参数默认值可能随代码版本改变，应以测试 commit 为准。

“参数支持验证”列依据同目录实测报告填写，每项结论链接到对应报告。`[√] 已支持` 仅表示报告所述版本、模型、配置与验证范围内的支持，不等于所有场景端到端通过；“需要开发支持”表示所测版本尚不可用；`[x] 无需支持` 表示字段已删除；`—` 表示本目录暂无该参数的独立报告。矩阵沿用原任务名，实际配置字段以报告为准。

| Args                           | Default value | Options                                           | UTs | supported | 参数支持验证 |
| ------------------------------ | ------------- | ------------------------------------------------- | --- | --------- | --- |
| model_path                                  | None          | str                                               |     | [√]        | — |
| config                                           | None          | str                                               |     | [√]        | — |
| text_only                                       | FALSE         | bool                                              |     | [√]        | — |
| colocate                                        | FALSE         | bool                                              |     | [x]       | — |
| isolate_stage                                 | None          | list[str]                                         |     |           | — |
| stage_process                               | None          | list[str]                                         |     |           | — |
| host                                              | 0.0.0.0         | str                                               |     | [√]        | — |
| port                                              | 8000           | int                                               |     | [√]        | — |
| model_name                                | None          | str                                               |     | [√] | — |
| allowed_local_media_path           | None          | str                                               |     | [√] | — |
| allowed_media_domain               | None          | list[str]                                         |     |           | [√] 已支持；Fun-CosyVoice3 域名允许/拒绝、默认行为及拒绝后合法音频生成通过（[报告](allowed_media_domain.md)） |
| tts_batch_max_items                    | 32               | int                                               |     |           | [√] 已支持；Fun-CosyVoice3 自定义上限 2/3、默认上限 32 的数量边界与合法批量生成通过，不代表音频逐字正确（[报告](tts_batch_max_items.md)） |
| mem_fraction_static                     | None          | float                                             |     |           | [√] 已支持；Fun-CosyVoice3 关闭图执行时，默认及显式比例的 KV 分配与音频生成通过，非进程总内存硬上限（[报告](mem_fraction_static.md)） |
| thinker_mem_fraction_static        | None          | float                                             |     |           | [√] 已支持；Qwen3-Omni 关闭图执行、TP=2 时，两个 Thinker 进程的 KV 分配及文字/音频生成通过（实际入口 `--thinker.engine.mem_fraction_static`）（[报告](thinker_mem_fraction_static.md)） |
| talker_mem_fraction_static          | None          | float                                             |     |           | [√] 已支持；Qwen3-Omni 关闭图执行时，Talker KV 分配与音频生成通过，共卡可用内存仍影响容量（实际入口 `--talker_ar.engine.mem_fraction_static`）（[报告](talker_mem_fraction_static.md)） |
| encoder_mem_reserve                | None          | float                                             |     |           | |
| cpu_offload_gb                           | None          | int                                               |     |           |  |
| quantization                                | None          | str                                               |     | [x]       | — |
| log_level                                      | "info"   | ["debug", "info", "warning", "error", "critical"] |     | [√]        | — |
| thinker_tp_size                            | None          | int                                               |     | [√]        | — |
| thinker_gpus                               | None          | str                                               |     | [√]        | — |
| image_encoder_tp_size               | None          | int                                               |     |           | — |
| image_encoder_gpus                  | None          | str                                               |     |           | [√] 已支持；Ming 单卡选卡与图片问答闭环通过（实际入口 `--image_encoder.gpu`）（[报告](image_encoder_gpus.md)） |
| talker_gpu                                   | None          | int                                               |     | [√]        | — |
| code2wav_gpu                            | None          | int                                               |     | [√]        | — |
| thinker_cuda_graph                    | "default"     | default\|on\|off                                  |     | [√]        | — |
| talker_cuda_graph                      | "default"     | default\|on\|off                                  |     | [x]       | — |
| talker_partial_start                      | "default"     | default\|on\|off                                  |     |           | [√] 已支持；实际字段 `enable_partial_start`，第 5 个片段即启动，开关对照与音频输出通过（[报告](talker_partial_start.md)） |
| thinker_torch_compile               | "default"     | default\|on\|off                                  |     |           | 需要开发支持；配置可传递，所测 NPU 编译路径未通过（[报告](thinker_torch_compile.md)） |
| talker_torch_compile                  | "default"     | default\|on\|off                                  |     |           | 需要开发支持；配置可传递，所测 NPU 编译路径未通过（[报告](talker_torch_compile.md)） |
| thinker_torch_compile_max_bs   | None          | int                                               |     |           | 需要开发支持；依赖的编译路径未通过，批大小边界无法验证（[报告](thinker_torch_compile_max_bs.md)） |
| talker_torch_compile_max_bs     | None          | int                                               |     |           | 需要开发支持；依赖的编译路径未通过，批大小边界无法验证（[报告](talker_torch_compile_max_bs.md)） |
| torch_compile                             | "default"     | default\|on\|off                                  |     |           | 需要开发支持；广播配置可传递，所测 NPU 编译路径未通过（[报告](torch_compile.md)） |
| torch_compile_max_bs                | None          | int                                               |     |           | 需要开发支持；广播配置可传递，依赖的编译路径未通过（[报告](torch_compile_max_bs.md)） |
| enable_realtime                           | FALSE         | bool                                              |     |           | — |
| decode_mode                              | None          | str                                               |     |           | [x] 无需支持；字段已删除，现行配置明确拒绝旧字段（[报告](decode_mode.md)） |
| async_lookahead_min_batch_size | None          | int                                               |     |           | [√] 已支持对应门槛功能；实测字段为 `async_decode_min_batch_size`，不代表原任务名可用于配置，未证实稳定加速（[报告](async_lookahead_min_batch_size.md)） |
| thinker_max_running_requests     | None          | int                                               |     |           | [√] 已支持；thinker 超限排队与阶段隔离通过（[报告](thinker_max_running_requests.md)） |
| prefill_coalesce_requests              | None          | int                                               |     |           | [√] 已支持；TP=1 合批与阈值边界通过，TP>1 时强制清零（[报告](prefill_coalesce_requests.md)） |
| prefill_coalesce_wait_ms               | None          | float                                             |     |           | [√] 已支持；TP=1 配对凑批参数验证到期放行，TP>1 时不参与判断（[报告](prefill_coalesce_wait_ms.md)） |
| max_running_requests                  | None          | int                                               |     |           | [√] 已支持；thinker / talker_ar 限流、超限排队与阶段隔离通过（[报告](max_running_requests.md)） |
| max_queued_requests                  | None          | int                                               |     |           | [√] 已支持；超限拒绝与恢复通过，chat 接口返回 500，单独设置时限流层级与语义不同（[报告](max_queued_requests.md)） |
| max_total_tokens                         | None          | int                                               |     |           | [√] 已支持；KV 池容量与准入生效，存在错误提示不准（[报告](max_total_tokens.md)） |
| cuda_graph_max_bs                    | None          | int                                               |     | [√]        | — |

## Issue 中已有的验证反馈

[2026-09-07 的 allowed_local_media_path 反馈](https://github.com/sgl-project/sglang-omni/issues/1597#issuecomment-5564446769)报告：参数路径校验逻辑未发现问题，但在 NPU 上为 Qwen3-TTS 输入 WAV 时遇到 kernel 编译失败，没有音频输出。

该评论将原因归于 NPU 平台集成，但未给出完整环境、启动命令和错误日志，本文未独立确认根因。此证据不能认定该场景端到端通过，也不能仅凭模型执行失败认定参数本身不受支持。

## 后续验收要求

每份实测报告应提供模型与代码版本、软件硬件环境、参数取值、可复现命令、预期行为、实际结果及证据。分别记录参数解析、配置传递、实际功能及端到端结果。失败时标明是参数本身、模型算子、资源约束还是尚未定位。
