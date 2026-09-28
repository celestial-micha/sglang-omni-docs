# 凑批最多压多久：prefill_coalesce_wait_ms

本次任务是检查：换用 NPU（昇腾计算设备）后，"为了凑一批最多愿意等多久"这个设置能否正常工作。

## 1. 验证对象与结论

上一份的 `prefill_coalesce_requests` 决定"凑够几条才发"，本条决定**"最多愿意为凑批等多久"**。两者是一对：只设条数不设时长，服务可能一直等下去，所以时长是**兜底**——等够了就先发，不再等。

它是**上限，不是固定延迟**：排队数在到期之前先凑够了，就会提前放行，不用等满。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `prefill_coalesce_wait_ms` | 需要，与凑批条数配对使用 | **[√] 已支持**。与上一份**同生共死**：它是同一段代码里的配对参数，在 `tp_size = 1` 的单卡模型上实测到期边界成立（配置 300 ms：压住时等待时长最高 240.46 ms，超过 300 ms 后的下一次判定即放行）。**使用说明**（与"是否支持"无关）：`tp_size > 1` 时该值虽被存进调度器，却**没有任何读取点**，形成"看起来生效了"的假象。 |

## 2. 测试用例

### 2.1 环境信息

与上一份 `prefill_coalesce_requests` **同一次实验、同一组配置**（这两个参数是一对，共用同一批数据，不是两次独立实验）。

| 项目 | 信息 |
| --- | --- |
| 硬件 | 昇腾 A3（Ascend910，64GB HBM ×16 chip），主机 `113.46.21.151` |
| 软件 | CANN 9.1 + SGLang 0.5.19 + sglang_omni 0.1.6 + transformers 5.12.1 + torch_npu 2.10.0 |
| 模型 A（TP=2，失效场景） | Qwen3-Omni-30B-A3B-Instruct（**必须 TP=2**） |
| 模型 B（TP=1，生效场景） | Qwen3-TTS-12Hz-0.6B-CustomVoice（**单卡即可**） |
| 容器 | `sglang-omni:pr2082-139a57e7-a3`；生效场景用**仅加日志**的 `sglang-omni-verify:probe3` |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 取值守卫 | 设 `wait_ms = 0`，启动 | 应被拒绝（`gt=0`） | 通过：`config/schema.py:240` 声明 `gt=0`，0 被拒 | — |
| **TP=2：值落地但无读取点** | Qwen3-Omni（TP=2）设 `wait_ms: 300`，探针读取调度器字段 | 值应为 300，并参与到期判断 | **值落地（300.0）但从不参与**：`coalesce_requests` 被清零后闸门第一道门即短路，`prefill_coalesce_wait_s` **没有任何读取点** | 见下的"半生效假象" |
| TP=2：到期判断是否发生 | 检查是否有任何一次因到期而放行的记录 | 应有 | **无**：闸门从未进入到期判断（82 次采样 `held_off` 全为 false） | — |
| **TP=1：值落地并参与判断** | Qwen3-TTS（`tp_size: 1`）设 `wait_ms: 300`，探针读取 | 值应为 300，且到期即放行 | 通过：`coalesce_wait_ms = 300.0`，**无 disabled 告警** | — |
| **TP=1：到期边界准确** | 先起 1 条长请求进入 decode，2 s 后发 1 条（排队数 1 < 阈值 4），记录每次判定的等待时长 | 等待到达 300 ms 后放行 | 通过：`oldest_wait_ms` = **0.07 → 120.45 → 240.46**（逐级上升，**均 < 300**），下一次判定即放行，该次调用 `call_ms=1536`（真实执行了 prefill） | **判定时刻不可直接观测**（探针时间戳记在调用结束时，而放行那次调用本身跑了 1536 ms）。能确证的是"压住时 ≤ 240.46 ms、下次判定时已 > 300 ms"，**300 ms 是上界** |
| **TP=1：是上界而非固定延迟** | 先起 1 条长请求使其进入 decode，再**同时**提交 4 条（`burst`） | 凑够就提前放行，**不等满 300 ms** | 通过：**3 次**采样均为 `waiting=4, held_off=false`，`oldest_wait_ms` = **2.08 / 2.16 / 2.17**（远低于 300），`prefill_bs=3` | 这是本参数"上限"语义的**直接证据**；逐条间隔到达不易复现，burst 可稳定复现 |
| ON/OFF 对照（隔离本参数） | 只把 `prefill_coalesce_requests` 由 4 改为 0，`wait_ms` 始终为 300 | OFF 组应完全不压 | 通过：两组 `coalesce_wait_ms` **均为 300.0**，但 OFF 组压住 **0 次**、ON 组压住 **4 次** | **本参数单独不产生行为**——它必须与条数配对才有意义 |

**"半生效假象"是什么**

`sglang_omni/scheduling/omni_scheduler.py:305-316` 里，TP>1 时**只清 `requests`，不清 `wait_ms`**：

```python
if requests > 1 and int(get_parallel().tp_size) > 1:
    logger.warning("Prefill admission coalescing is disabled for tp_size=...")
    requests = 0                                   # ← 只清了这一个
self.prefill_coalesce_requests = requests
self.prefill_coalesce_wait_s = wait_ms / 1e3       # ← 照常赋值，永不被读取
```

后果：任何"配置是否被接受""参数是否已生效"的检查**都会显示正常**（字段里有 300.0），但功能实际是关的。加上告警只打一次、级别为 WARNING、混在几百行启动日志里，极易漏掉。

**建议**（供修复参考）：TP 禁用时**一并把 `wait_s` 归零或置空**，避免留下一个"看似有效"的数值；或在配置校验阶段就把"`requests ≥ 2` 且 `tp_size > 1`"这种**必然不生效**的组合判为错误或 ERROR 级提示。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 取值合法性（`gt=0`，0 被拒） | ✓ `pipeline/test_runtime_schema.py:75`（`wait_ms=300` 接受、`0.0` 抛错） | ✗ |
| 归一化（毫秒 → 秒，0.06 / 0.3） | ✓ `pipeline/test_scheduler.py:2559` | ✗ |
| **到期放行分支** | ✓ `scheduling/test_prefill_coalesce_gate.py::test_small_queue_is_held_until_oldest_expires`、`::test_pending_build_gate_releases_at_deadline` | ✓ 实测（等待 240.46 ms 后下一次判定放行） |
| **"上界而非固定延迟"语义** | ✓ 同文件 `::test_reaching_target_releases_before_deadline` | ✓ 实测（3 次 `oldest_wait`=2.08/2.16/2.17 ms 即放行） |
| 计时戳跨部分准入/中止仍保持 | ✓ 同文件 `::test_partial_admission_leftovers_keep_their_deadline`、`::test_abort_does_not_hand_newcomers_an_expired_deadline` | ✗ |
| **`tp_size > 1` 时 `wait_s` 残留非零值** | ✗ **无覆盖** | ✓ 实测（300.0 落地但从不参与） |

**关于 UT：到期判定与"上界"语义覆盖得较完整**，`test_prefill_coalesce_gate.py` 用可控的假时钟覆盖了到期、提前放行、计时戳保持等分支。

**缺口与上一份同源**：让本参数在 TP>1 下失效的那条规则（以及它留下的 `wait_s` 残留）**没有测试** —— 测试桩绕过了 `__init__`，而清零与赋值都写在 `__init__` 里。UT 也**无法发现"字段里有值但没有任何读取点"**这类问题：它验的是"值写对了没有"，而这里的问题恰恰是"值写对了，但没人读"。
