# 输入还没到齐就抢先开始生成语音：talker_partial_start

本次任务是检查：换用 NPU（昇腾计算设备）后，"语音阶段在文字还没全部产出时就先开工"这个设置能否正常工作。

> **参数名提示**：`talker_partial_start` 是**历史名称**（曾是一个命令行开关 `--talker-partial-start`），在 sglang_omni **v0.1.4 起已被删除**。当前版本（0.1.6）的**真实参数**是：
>
> | 真实名称 | 位置 | 默认值 |
> | --- | --- | --- |
> | `talker_ar.factory.enable_partial_start` | `models/qwen3_omni/config.py:209,228` | 非共置布局 **True**；共置布局 **False** |
> | `talker_ar.factory.partial_start_min_chunks` | 同上 `:229` | **5** |
>
> 按旧名配置**不会报"未知参数"，而是静默不生效**。本报告按真名验证。

## 1. 验证对象与结论

一个语音请求里，负责理解与文字的"thinker"阶段先把内容产出，负责声音的"talker"阶段再据此生成语音。传统做法是**等文字全部产出**，talker 才开始。

`enable_partial_start` 允许 talker **在文字还没全部产出时就提前开工**，只要已经攒够一定数量的片段（由 `partial_start_min_chunks` 决定，默认 5）。好处是**降低首包延迟**——不必等全文结束。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `talker_partial_start`（真名 `enable_partial_start`） | 需要，属 omni 自有调度优化，与设备无关 | **[√] 已支持**。开关因果明确、边界精确：开启时**第 5 个片段、且输入流仍在进行**（`stream_done = False`）即开工；关闭时需等满 **148** 个片段且流结束。**这是 5 个参数中边界被完整覆盖、且行为完全符合设计的一个。** |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | 昇腾 A3（Ascend910，64GB HBM ×16 chip），主机 `113.46.21.151`，本次用 11、12、13 号卡 |
| 软件 | CANN 9.1 + SGLang 0.5.19 + sglang_omni 0.1.6 + transformers 5.12.1 + torch_npu 2.10.0 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（语音管线 7 个 stage：thinker **TP=2**（11、12 号卡）+ `talker_ar` + `code2wav`（13 号卡）） |
| 容器 | `sglang-omni-verify:probe3`（= `sglang-omni:pr2082-139a57e7-a3` + **仅加日志**的探针，不改行为） |

> `talker_scheduler.py` 中**没有** `tp_size` 门控，因此本参数可以在 TP=2 下正常验证 —— 这与 `prefill_coalesce_*` 形成鲜明对比。

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 参数真名核对 | 全仓库检索 `talker_partial_start` | 应存在 | **不存在**（零命中）。真名为 `talker_ar.factory.enable_partial_start` | 旧名是 v0.1.3 及以前的命令行开关，v0.1.4 配置重构时删除 |
| 开关值传入正确位置 | ON / OFF 两份配置各启动一次，探针读取 talker 调度器字段 | 应分别读到 True / False | 通过：ON 读到 `True`（18/18），OFF 读到 `False`（6/6） | — |
| 阈值参数正确 | 读取 `partial_start_min_chunks` | 应为 5 | 通过：18/18 采样均为 **5**，与 `config.py:229` 一致 | — |
| 走哪条判定路径 | 读取 `talker_start_topology` | —— | 恒为 `False` → 走 legacy 路径，阈值取 `partial_start_min_chunks`（5），非新拓扑的 `TALKER_START_MIN_CHUNKS`（1） | — |
| **ON：未到齐即开工** | 提交语音请求（`modalities: ["text","audio"]`），记录每次就绪判定的 `usable_chunks / ready / stream_done` | 片段数达到阈值且流未结束时 `ready` 变为 True | 通过：序列 `0→F 1→F 2→F 3→F 4→F`，**`5→True（stream_done=False）`**，3/3 次请求一致 | — |
| **OFF：必须等到齐（对照）** | 同一模型、同一请求文本、同一会话，仅把开关关掉 | `ready` 只应在 `stream_done=True` 时出现 | 通过：序列 `0→F`，**`148→True（stream_done=True）`**；"未到齐即开工"出现 **0 次** | — |
| 阈值精度 | 检查 `ready` 翻转点 | 应恰好在 5，而非 4 或 6 | 通过：3 次请求均在 `usable_chunks = 5` 翻转 | — |
| 判定频次差异（机制旁证） | 统计探针被调用次数 | ON 应显著多于 OFF | 通过：ON 18 行 vs OFF 6 行 | 因 `_should_recheck_deferred_request_on_stream_chunk()` 直接返回 `self._enable_partial_start`（`talker_scheduler.py:124-128`）：ON 时每个片段重判，OFF 时只在开头与流结束各判一次 |
| 音频产出 | 两组各提交 2 次请求，保存返回音频 | 应有音频文件且请求成功 | 通过：全部 HTTP 200；ON：1 826 774 B / 1 730 774 B；OFF：1 703 894 B / 1 869 014 B | — |
| 文本输出 | 读取回复正文 | 应为完整故事段落 | 通过：两组均返回完整故事文本 | — |

**A/B 对照汇总**

| | ON | OFF |
| --- | --- | --- |
| 探针行数 | 18 | 6 |
| `(ready, stream_done)` 分布 | `{(False,False): 15, (True,False): 3}` | `{(False,False): 3, (True,True): 3}` |
| **`ready=True` 且 `stream_done=False`（真正的部分启动）** | **3 次** | **0 次** |
| 开工前需到齐的片段数 | **5** | **148** |

**开关把"开工前必须到齐的输入量"从 148 个片段降到 5 个，相差约 30 倍。** 3 组请求（1 次预热 + 2 次实测）序列完全一致，两个方向都不是噪声。

**为什么这条对照是必需的**：该参数**默认值本来就是 `True`**。原报告"开关传到了正确位置…接受 True"这一条，**无法区分"开关生效"与"只是默认值"**。只有 ON/OFF 两侧都测，才能把因果钉死。

**判定逻辑**（`talker_scheduler.py:102-119`）：

```python
def _is_request_build_ready(self, payload, *, pending_stream_done) -> bool:
    if pending_stream_done:
        return True                      # 全部到齐 -> 必然就绪
    if not self._enable_partial_start:
        return False                     # 开关关闭 -> 必须等到齐
    prefetched = getattr(payload, "prefetched_chunks", None) or []
    usable = self._count_usable_prefetched_chunks(prefetched)
    if self._talker_start_topology:
        return usable >= TALKER_START_MIN_CHUNKS        # 新拓扑（本次未启用）
    return usable >= self._partial_start_min_chunks     # legacy，= 5
```

约束：`partial_start_min_chunks >= MIN_PARTIAL_START_CHUNKS = 3`，否则 `talker_scheduler.py:68` 抛 `ValueError`。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 配置路由（点号路径开/关） | ✓ `qwen3_omni/test_cli.py:327,334,350,358`（默认值、置 false/true、非布尔被拒、stage 名写错时报错） | ✓ ON/OFF 两次启动 |
| 默认值（含共置布局差异） | ✓ `qwen3_omni/test_example_launcher.py:395,406,416,426,436,446` | ✓ |
| 下限约束（低于 `MIN_PARTIAL_START_CHUNKS` 被拒） | ✓ 同文件 `:459` | ✗ |
| **开关关闭时即使有 10 个片段也不就绪** | ✓ `qwen3_omni/test_talker.py:2639`（参数化 `topology=[False, True]`，并断言重判函数返回 False） | ✓ OFF 组 0 次提前开工 |
| 开关开启且片段不足时**不就绪**（反例） | ✓ 同文件 `:2650`（`min_chunks=5`，1 个片段 → 不就绪） | ✓ ON 组 `0..4 → False` |
| **开关开启且片段达到阈值时"就绪"（正例）** | ✗ 只断言了反例，"5 个片段且流未结束 → 就绪"这一**肯定分支没有断言** | ✓ 实测 `5→True(stream_done=False)`，**这才是参数生效的关键路径** |
| `_count_usable_prefetched_chunks` 的 `im_end` 扣减 | ✗ 最后一个片段的 `token_id == im_end_token_id` 时少算 1，该分支无覆盖 | ✗ |
| 新拓扑（`TALKER_START_MIN_CHUNKS = 1`）路径 | ✗ 现有用例只覆盖 legacy 路径 | ✗ 本次实测走的也是 legacy |
| 真实分片时序（"第 5 片到齐时流仍在进行"） | ✗ 不适合单测 | ✓ 本次实测正是补齐这一层 |

**这是 5 个参数中 UT 覆盖最好的一组**：配置路由、默认值（含共置差异）、下限约束、以及调度器的判定函数均有覆盖。原报告只引用了 `test_cli.py` 与 `test_example_launcher.py`，**漏掉了 `test_talker.py` 中两个真正验证判定逻辑的行为用例**。

**建议补的用例**：阈值正例（5 个片段且流未结束 → 就绪）、阈值边界成对（4 → False、5 → True）、`im_end` 扣减、新拓扑路径（1 个片段即就绪）、到齐优先（开关关闭但 `pending_stream_done=True` → 就绪）。
