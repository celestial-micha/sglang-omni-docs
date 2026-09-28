# 限制模型计算缓存（KV 池）的大小：max_total_tokens

本次任务是检查：换用 NPU（昇腾计算设备）后，"把模型的计算缓存手动定成多大"这个设置能否正常工作。

## 1. 验证对象与结论

模型边算边记：已经处理过的内容存进一块缓存，后续计算直接取用，不用重算。这块缓存叫 **KV 池**。不指定大小时，程序按"留给模型的显存比例"自己算；`max_total_tokens` 就是**手动把这块池子钉成一个固定值**（单位是 token 数）。

**要点：它是"整块池子有多大"，不是"每条请求能用多少"。** 池子是**所有并发请求共用**的，谁先到谁先占，占满就轮不到别人。上游 SGLang 对这个参数的说明也写着它"typically used for **development and debugging purposes**"（一般用于开发调试）。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `max_total_tokens` | 需要，用于精确控制缓存占用 | **[√] 已支持**。设定的值精确落地，池满时的准入行为正确。但有两处问题：**报错信息指向了一个改不动这件事的参数**；**池子小于上下文长度时配置期完全没有校验**。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | 昇腾 A3（Ascend910，64GB HBM ×16 chip），主机 `113.46.21.151` |
| 软件 | CANN 9.1 + SGLang 0.5.19 + sglang_omni 0.1.6 + transformers 5.12.1 + torch_npu 2.10.0 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（thinker **TP=2**） |
| 容器 | `sglang-omni:pr2082-139a57e7-a3`，本次用 8、9、10 号卡 |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 值精确落地 | 设 `max_total_tokens: 4096`，启动后读分配日志 | 日志中的池容量等于 4096 | 通过：`KV Cache is allocated ... #tokens: 4096`，K/V 各 0.10 GB | 两个 rank 各一份，**不能相加** |
| 总量超池应放行 | 12 条**唯一 prompt**（每条约 1792 token，合计 20333）并发提交，池仅 4096 | 请求全部成功（排队串行），不因容量不足报错 | 通过：`12 × HTTP 200`，wall 1.29 s | 不设本参数时同一配置池为 **498304** token |
| 超池时的调度机制 | 观察批次统计与抢占计数 | 应通过准入控制逐条放行 | 通过：`#running-req` **恒为 1**（`max_running_requests` 默认 64），`token usage` 稳在 0.44，**retract/preempt 计数 = 0** | **是串行放行，吞吐并未提高** |
| 单请求接近池容量 | 单条 prompt 约 3724 token（< 4096） | 成功 | 通过：HTTP 200 | — |
| 单请求超池容量 | 单条 prompt 约 4956 / 6188 token（> 4096） | 被拒绝 | 通过：HTTP 400，`kv_capacity=4095` | — |
| 错误提示的准确性 | 阅读 400 的提示文本 | 应指向造成限制的参数（`max_total_tokens`） | **不符**：提示为 `Current mem_fraction_static is 0.850; try setting --thinker-mem-fraction-static higher.` | 池被 `max_total_tokens` 钉死时**调它完全无效**，见下 |
| 池子/上下文缝隙 | 提交 prompt 长度介于 4096 与 8192 之间的请求 | 配置期应给出校验或提示 | **无任何校验**，请求期一律 400 | 该区间**能通过配置校验但必然失败** |

**"调 mem_fraction_static 没用"的代码依据**

`mem_fraction_static` 是"显存里预留给静态部分（模型权重 + KV 池）的比例"，调大它确实能让**按内存算出来的**池容量变大。但上游把它当**上限**用：

```python
sglang/srt/mem_cache/kv_cache_configurator.py:302-305
token_capacity = min(token_capacity, max_total_tokens)
```

是 `min`，所以池被 `max_total_tokens` 钉在 4096 时，把 `mem_fraction_static` 调大只会抬高 `token_capacity`，`min` 之后**仍然是 4096** —— 那句报错建议让人做无用功。

另外上游的校验是**单向的**：池配得比实测容量**大**会 warning（`kv_cache_configurator.py:2270-2275`），配**小**则完全不提示。

**"池子与上下文的关系"是怎么处理的（不是冲突，是取小）**

```python
omni_scheduler.py:332-335
self.max_req_len = min(
    server_args.context_length - 1,
    effective_max_total_num_tokens - 1,
)
```

所以池 4096 + 上下文 8192 → 单请求上限 = `min(8191, 4095)` = **4095**，与实测 `kv_capacity=4095` 逐位吻合。`model_runner/model_worker.py:196-201` 用的是**完全相同**的公式（两处一致）。

**代价**：`context_length: 8192` 被**静默架空** —— 有效期其实只有 4096，配置里那句话成了摆设，而配置期没有任何提示。

**"单请求能独占整个池"（本参数最需要注意的使用陷阱）**

真正管准入的是 `PrefillAdder`（`sglang/srt/managers/schedule_policy.py:558`）：

```python
:596  self.memory_budget = token_to_kv_pool_allocator.create_prefill_budget(...)   # 共享预算
:711  def rem_total_tokens(self):
:712      return self.memory_budget.remaining_total                                # 池子还剩多少
```

每接纳一条就扣减，**没有任何按请求等分的逻辑**。而唯一的"单请求"上限恰恰**是从整个池子推出来的**（`max_req_len = min(context_length, 池) - 1`）。所以：

> **一条请求只要 `输入 + 最大生成长度 ≤ 池 − 1` 就合法 —— 也就是单请求可以吃掉几乎整个池，后面的请求全部饿死。**

对照上一行"总量超池应放行"的实测数据（`#running-req` 恒为 1、`token usage` 稳在 0.44）就是这件事的表现：池子是共享的，占满只能串行。

**由此得到的使用结论**：`max_total_tokens` **不能当"单请求预算"用**。把它设成 4096 不等于"每条请求 4096"。**想限制单条请求的占用，应该调 `context_length` 或 `max_new_tokens`，而不是调 `max_total_tokens`。**

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 与 `kv_cache_bytes` 互斥 | ✓ `pipeline/test_runtime_schema.py:203`（同时设抛 `ValueError`） | ✗ |
| 传入 ServerArgs 的合并/剔除 | ✓ `serve/test_generation_server_args.py:35,92` | ✗ |
| **取值下界（`ge=1`，设 0 应被拒）** | ✗ `config/schema.py:156` 声明了 `ge=1`，但**没有用例断言"设为 0 会被拒绝"** | ✗ |
| **池满时的准入行为（逐条放行、不抢占）** | ✗ | ✓ 实测 `#running-req` 恒 1、retract/preempt = 0 |
| **单请求超出池容量的处理** | ✗ 返回值（400）与消息模板均无断言 | ✓ 实测 400 |
| **单请求"占满"池对并发的影响** | ✗ | ✗ **本次也未做** —— 这是已知缺口（见下） |
| **400 提示里该出现哪个参数名** | ✗ 无任何断言 | ✓ 实测提示指向了错误参数 |
| 池 < 上下文时的配置校验 | ✗ 无校验、也无 UT 固定该行为 | ✓ 实测该区间一律 400 |

UT 集中在"值能不能正确写进去"和"与另一个参数互斥"，**行为层（池满怎么调度、单条超池怎么拒、提示说什么）全部没有覆盖**，因此那句错误提示长期未被发现。

**已知缺口**：本次**没有**测"一条贪心长请求（占满池）＋ 若干短并发"这个场景。它是"池子是共享的、单请求上限却等于整个池"这一结论的**直接验证**，建议补测：
- 池设 4096；先发 1 条约 4000 token 的长请求，再并发 3 条短请求 → 断言短请求被**完全阻塞**到长请求结束、`token usage` 长期贴顶、`#running-req` 恒为 1。
