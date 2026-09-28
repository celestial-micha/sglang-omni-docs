# 凑够几条请求再一起送进计算：prefill_coalesce_requests

本次任务是检查：换用 NPU（昇腾计算设备）后，"让服务等一等、凑几条请求一起算"这个设置能否正常工作。

## 1. 验证对象与结论

请求进入计算前有一个"预处理（prefill）"步骤。这个步骤**每次启动都有固定开销**，所以一次处理 4 条和一次处理 1 条，开销差不多。

`prefill_coalesce_requests` 就是**"凑够几条再一起发"**：设为 4，服务发现排队区还不够 4 条时，**会故意先压住不发**，等凑够 4 条（或等到超过 `prefill_coalesce_wait_ms` 规定的时限）再一起送进计算。

**它不是一个"制造批次"的开关** —— 上游本来就会把同时可用的请求合批。它的作用是**"即使有请求在等，也故意先不发"**，用一点延迟换更饱满的批次。因此判据是**多出来的那段延迟**，看"批次里有 4 条"会得出错误结论（空闲时同样会合批，且没有任何额外延迟）。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `prefill_coalesce_requests` | 需要，属于调度优化，与设备无关 | **[√] 已支持**。在 `tp_size = 1` 的单卡模型（Qwen3-TTS-12Hz-0.6B）上实测生效，边界两侧都验到。**使用说明**（与"是否支持"无关）：代码对 `tp_size > 1` 有一条硬规则——该值会被强制清零，所以对必须 TP=2 的大模型（如 Qwen3-Omni-30B）这个参数不会被采用。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | 昇腾 A3（Ascend910，64GB HBM ×16 chip），主机 `113.46.21.151` |
| 软件 | CANN 9.1 + SGLang 0.5.19 + sglang_omni 0.1.6 + transformers 5.12.1 + torch_npu 2.10.0 |
| 模型 A（TP=2，失效场景） | Qwen3-Omni-30B-A3B-Instruct（thinker 权重 63.44 GB > 单卡 61.27 GiB，**必须 TP=2**） |
| 模型 B（TP=1，生效场景） | Qwen3-TTS-12Hz-0.6B-CustomVoice（4.3 GB 量级，**单卡即可**） |
| 容器 | `sglang-omni:pr2082-139a57e7-a3`；参数无原生日志，生效场景另用**仅加日志**的派生镜像 `sglang-omni-verify:probe3` |

> 探针只包装调度器方法并原样返回其返回值，**不改变任何控制流**，且仅在设置环境变量时启用。

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| **TP=2：参数被清零** | Qwen3-Omni（thinker TP=2）设 `prefill_coalesce_requests: 4`，启动后由探针读取调度器实际字段 | 调度器中该值应为 4 | **不符：实际为 0**，82 次采样全部为 0；服务端打出 `Prefill admission coalescing is disabled for tp_size=2 ...` | 这是**设计性禁用**，不是观测问题（见下） |
| TP=2：是否参与调度 | 制造"decode 在飞 + 排队数 < 阈值"场景 | 应被压住 | **从未压住**（82 次采样 `held_off` 全为 false） | 清零后闸门第一道条件即短路 |
| **TP=1：配置值落地** | Qwen3-TTS（`tp_size: 1`）设 `prefill_coalesce_requests: 4`，探针读取 | 调度器中该值应为 4 | 通过：`coalesce_requests = 4`、`coalesce_wait_ms = 300.0`、`coalesce_when_idle = false`，**无 disabled 告警** | 与 TP=2 形成正面对照 |
| **TP=1：空闲时不压** | 空闲状态发 1 条 | 直接处理，不压 | 通过：`decode_idle=true → held_off=false`，直接 `prefill_bs=1` | — |
| **TP=1：未达阈值要压住** | 先起 1 条长请求使其进入 decode，2 s 后再发 1 条（排队数 1 < 阈值 4） | 该条被压住，直到凑够或到期 | 通过：连续 3 次采样 `held_off=true`，`oldest_wait_ms` = **0.07 → 120.45 → 240.46**（逐级上升） | 归档日志 580 条采样中共 4 次压住，`waiting` **均为 1** |
| **TP=1：到期放行** | 同上，观察何时放行 | 超过 `wait_ms`（300 ms）后放行 | 通过：压到 **240.46**（< 300）后，下一次判定即放行，该次调用 `call_ms=1536` —— **真实执行了 prefill** | **判定时刻不可直接观测**：探针的时间戳记在调用**结束时**，而放行那次调用本身跑了 1536 ms。能确证的只是"压住时 `oldest_wait` ≤ 240.46 ms、下次判定时已 > 300 ms 故放行"，**300 ms 是上界** |
| **TP=1：达阈值提前放行** | 先起 1 条长请求使其进入 decode，再**同时**提交 4 条（`burst`），让排队数一步升到 4 | 排队数一到 4 就立即放行，不必等满 300 ms | 通过：**3 次**采样均为 `waiting=4, held_off=false`，`oldest_wait_ms` = **2.08 / 2.16 / 2.17**（**远低于 300**），且每次都 `prefill_bs=3` —— **3 条进了同一个 prefill 批** | 直接观察到参数的**设计目的**：合批。注：用"逐条间隔到达"不易复现（本管线 preprocessing 串行，请求进等待队列偏慢），**用 burst 稳定复现** |
| **TP=1：阈值以上不再压** | 统计所有采样中压住时的排队数 | 排队数 ≥ 4 时不应再压 | 通过：580 条采样中压住 4 次，`waiting` **全部为 1**；`waiting = 4` 的 4 条采样**全部 `held_off=false`** | — |
| **ON/OFF 对照** | 除 `prefill_coalesce_requests`（4 → 0）外配置完全相同，各跑同样负载 | ON 有压住、OFF 无 | 通过：ON **压住 4 次** / OFF **压住 0 次**；两组 `coalesce_wait_ms` 均为 300.0（单独不产生行为）；两组均无 disabled 告警 | 唯一变量换来唯一差异，因果明确。两组日志各约 520 行，ON 580 条探针采样 / OFF 145 条 |

**为什么 TP=2 下是"架构性失效"而不是"测不出来"**

| 事实 | 数值 |
| --- | --- |
| Qwen3-Omni thinker bf16 权重 | **63.44 GB** |
| 单卡 `total_memory` | **61.27 GiB** |
| 结论 | 必须 **TP=2** |
| 禁用门槛 | `tp_size > 1` |

**触发条件与可用条件互为反面。** 而且 `/mnt/paas/weights` 下**没有**更小的 Qwen3-Omni 或量化版 —— 这就是本次改测 Qwen3-TTS 的原因。

**被清零的源码依据**（`sglang_omni/scheduling/omni_scheduler.py:305-316`，实测与部署镜像逐行一致）：

```python
requests = int(prefill_coalesce_requests)
wait_ms = float(prefill_coalesce_wait_ms)
if requests > 1 and int(get_parallel().tp_size) > 1:
    logger.warning(
        "Prefill admission coalescing is disabled for "
        f"tp_size={get_parallel().tp_size}: the wait deadline reads each "
        "rank's local clock, so ranks could disagree on expiry and "
        "break lockstep scheduling")
    requests = 0                                   # ← 强制清零
self.prefill_coalesce_requests = requests
self.prefill_coalesce_wait_s = wait_ms / 1e3       # ← wait_ms 照样存下来（见下一份报告）
```

清零后，`get_new_batch_prefill()` 第一道门即短路（同文件 `:1390`）：

```python
if self.prefill_coalesce_requests <= 1 or self.chunked_req is not None:
    return _Upstream.get_new_batch_prefill(self, running_batch)
```

**设计动机（为什么 TP 下不敢用）**：到期判断依赖进程内单调时钟，而 TP 下**每个 rank 是独立进程、各自完整地跑一遍调度决策**（跨 rank 广播的只有请求输入列表，**不是批决策本身**）。若两个 rank 对"是否到期"判断不一致，一个 rank 进入模型 forward 参加集合通信、另一个返回空批去休眠，就会**永久挂死**。同一个文件里还有一条**完全同构**的守卫：`request_build_max_workers` 在 `tp_size>1` 时也被压成 1，理由是"to preserve identical request admission order on every TP rank"。

> **一处措辞不精确（供参考）**：告警说"reads each rank's **local clock**"。Linux 上 CPython 的 `time.perf_counter()` 是同机 system-wide 的 `CLOCK_MONOTONIC`，且 `now` 与入队时间戳取自同一进程的同一时钟，**原点是一致的**；真正的分歧来源是**两个 rank 的采样时刻相差一个 δ**（广播 + 本地构造成本），使真实时长恰好落在 `wait_s ± δ` 窗口时两边答案不同。**结论（必须禁用）不变，但机制描述不够准确。** 此点属分析而非源码事实，未做实验证实。

**已知的观测陷阱（务必写进判据）**

参数**没有任何原生日志**，只能插桩。而探针里 `held_off = (批为空)` 这一判据**无法区分"闸门压住"和"上游自己拒绝准入"**：当 `running_batch_size` 已等于 `max_running_requests`（运行名额满）时，上游本来就返回空批。本次实测中**绝大多数** `held_off=true` 的采样属于后者（可连续 15 秒以上为 true，远超 300 ms 的到期线），**归档 ON 日志的 580 条采样里只有 4 条**是真正的闸门压住。正确判据是：

> **`held_off = true` 且 `running_batch_size < max_running_requests` → 上游本可接纳却返回空 → 只可能是闸门压住。**

不写这条判据会得出完全相反的结论。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 取值合法性（`ge=0`、`=1` 告警） | ✓ `pipeline/test_runtime_schema.py:75,85` | ✗ |
| 归一化（默认值、单位换算） | ✓ `pipeline/test_scheduler.py:2559,2572` | ✗ |
| **闸门压住分支** | ✓ `scheduling/test_prefill_coalesce_gate.py::test_small_queue_is_held_until_oldest_expires` | ✓ 实测 4 次 |
| **达阈值放行分支** | ✓ 同文件 `::test_reaching_target_releases_before_deadline` | ✓ 实测（`waiting=4`、`oldest_wait`=2.08/2.16/2.17 ms、`prefill_bs=3`） |
| **`chunked_req` 旁路** | ✓ 同文件 `::test_chunked_prefill_bypasses_gate` | ✗ |
| **`when_idle` 门控** | ✓ 同文件 `::test_idle_loop_can_coalesce_when_explicitly_enabled` | ✓ 实测（空闲直通） |
| 到期计时语义（`_coalesce_enqueue_t`） | ✓ 同文件 `::test_unstamped_request_is_stamped_and_eventually_released` 等 | ✓ 实测（0.07 → 120.45 → 240.46 → 放行） |
| **`tp_size > 1` 时被清零这条规则** | ✗ **无覆盖** | ✓ 实测（82 次采样全为 0 + 告警） |

**关于 UT：闸门本身的判定逻辑覆盖得很好。** `tests/unit_test/scheduling/test_prefill_coalesce_gate.py`（351 行 / 20 个用例，随功能同一个 commit 加入）覆盖了压住、达阈值放行、到期放行、`chunked_req` 旁路、空闲门控、计时戳、部分准入与中止等分支。`test_cli_prefill_coalesce.py` 另覆盖命令行点号路由。

**唯一缺口正是让本参数变死的那一条规则。** 原因很具体：该文件的测试桩 `_StubScheduler` **直接设置属性、绕过了 `OmniScheduler.__init__`**，而清零逻辑恰好写在 `__init__` 里（`:305-316`）。更讽刺的是 `test_scheduler.py:2572` 的 docstring 自称 *"the scheduler trusts its callers and keeps only its own **TP-interaction rule**"*，但它传入的是 `prefill_coalesce_requests=0` —— **连 `requests > 1` 这一半都是假的。它声称保留的那条规则，恰好是它唯一没测的那条。**

**文档层面**：`docs/cookbook/qwen3_tts.md` 有整节 "Prefill Admission Coalescing" 介绍这两个参数，**但通篇没有提到 `tp_size > 1` 会被禁用**。对 Qwen3-Omni 这类必须 TP=2 的模型，文档承诺了一个在该模型上**架构上不可能**的功能。
