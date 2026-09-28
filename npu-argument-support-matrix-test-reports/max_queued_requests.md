# 限制排队等待的请求数：max_queued_requests

本次任务是检查：换用 NPU（昇腾计算设备）后，"队伍里最多能排多少条请求"这个设置能否正常工作。

## 1. 验证对象与结论

一个服务同时收到很多任务时，一部分进设备计算，其余排队等候。`max_queued_requests` 就是**排队区能放多少条**：设为 2，服务最多收下 2 条在等，再多就直接拒绝（不让它进来白等）。NPU 上同样需要这个上限。

它和 `max_running_requests`（同时在算的名额）合起来决定服务总共收多少条：**能收下的总数 = 同时运行数 + 排队数**。

| 参数 | NPU 是否需要 | 本次结论 |
| --- | --- | --- |
| `max_queued_requests` | 需要，限制排队总量、防止无限堆积 | **[√] 已支持**。上限精确生效，超限正确拒绝、释放后能恢复。但有三处与预期不符：**返回 500 而不是文档说的 503**；实际拦截点在**管线协调器**而非引擎调度器；**只设它一个时语义会变**（详见 2.2 最后两行）。 |

## 2. 测试用例

### 2.1 环境信息

| 项目 | 信息 |
| --- | --- |
| 硬件 | 昇腾 A3（Ascend910，64GB HBM ×16 chip），主机 `113.46.21.151` |
| 软件 | CANN 9.1 + SGLang 0.5.19 + sglang_omni 0.1.6 + transformers 5.12.1 + torch_npu 2.10.0 |
| 模型 | Qwen3-Omni-30B-A3B-Instruct（thinker 权重 63.44 GB，单卡 61.27 GiB 装不下，**必须 TP=2**） |
| 容器 | `sglang-omni:pr2082-139a57e7-a3`，本次用 8、9、10 号卡 |

### 2.2 测试用例结果

| 用例名称 | 测试步骤 | 预期结果 | 实际结果 | 备注 |
| --- | --- | --- | --- | --- |
| 配置传入并生效 | 设 `max_running_requests: 1`、`max_queued_requests: 2`，启动后读日志 | 两个值被接受，总数上限生效 | 通过：日志 `Coordinator in-flight cap=3 (generation running+queued)`，3 = 1 + 2 算术吻合 | 上限在**启动时**就算好并打印，可核对 |
| 未达上限不拒绝 | 同时提交 3 条长任务 | 3 条全被接收 | 通过：`[200, 200, 200]`，耗时 11.6 / 19.8 / 28.1 s（严格串行，印证 running=1） | — |
| 超限 1 条被拒 | 同上，同时提交 4 条 | 第 4 条被拒 | 通过：`[200, 200, 200, 500]`，响应体 `{"detail":"The request queue is full."}`，拒绝耗时 0.004 s | **边界精确落在 3** |
| 超限 2 条被拒 | 同上，同时提交 5 条 | 后 2 条被拒 | 通过：`[200, 200, 200, 500, 500]` | — |
| 拒绝者归属 | 在服务端日志中检索三处拒绝点 | —— | 协调器 **3 次**（`before pipeline submit: in-flight cap`）；引擎调度器两处均为 **0 次** | 拦截发生在**比引擎更上一层**的管线协调器 |
| 拒绝后的恢复 | 上一轮出现 500 后，重新排空再提交一批 | 服务仍能正常处理 | 通过：下一轮仍拿到 `[200, 200, 200]`，未卡死 | — |
| 超限状态码 | 读取被拒请求的 HTTP 状态码 | 文档（`admission.py:8` docstring）声明 **503** | **不符：实测 500** | `/v1/chat/completions` 走 `serve/openai_api.py` 的 catch-all `HTTPException(status_code=500)`，未做 `QueueFullError → 503` 映射；**语音接口是对的**（`serve/speech_errors.py:127` 确实映射 503） |
| 只设排队上限时的语义 | 配置**只**设 `max_queued_requests: 2`（不设 `max_running_requests`） | 应仍按"队列长度 2"拦截 | 协调器上限**整个消失**（日志无 `in-flight cap`），改由引擎调度器拒绝（`before build`，6 次）。**同一参数含义变了** | 见下 |

**关于最后一行（语义随另一参数变化）**

协调器上限的计算要求**两个键都在**（`pipeline/mp_runner.py:68-80`）：缺任何一个就 `KeyError` → 整个上限关掉。而 `PipelineConfig.generation_admission_defaults()` 返回 `{}`（`config/schema.py:822-824`），**Qwen3-Omni 没有覆写它**，所以默认两个键都不在。

| 配置 | 协调器上限 | 实际生效的拒绝点 | `max_queued_requests` 的含义 |
| --- | --- | --- | --- |
| 两个都设 | `running + queued` | 协调器 | **管线级在飞总量**（跨全部 stage） |
| 只设 queued | 不存在 | 引擎调度器 | **引擎级等待队列长度** |

对照：**Qwen3-TTS 覆写了这个钩子**（`models/qwen3_tts/config.py:43-47`，给出 `max_running_requests: 16, max_queued_requests: 16`），所以在 TTS 上在飞上限**默认一直是开着的**。也就是说"必须两个都设"不是文档要求，是 **Qwen3-Omni 这个模型族没接上这个钩子**，而且只设一个时**静默降级、没有任何提示**。

> 补充：`max_queued_requests` 的上游定义里有一句"disaggregation-mode 下忽略此参数"，但 sglang-omni 的调度器 `disaggregation_mode` 恒为 `NULL`（`omni_scheduler.py:522-525`），**不适用**。

## 3. UT 覆盖

### 3.1 当前覆盖情况

| 覆盖场景 | UT 是否覆盖 | 端到端是否覆盖 |
| --- | --- | --- |
| 调度器侧"构建前拒绝" | ✓ `pipeline/test_scheduler.py::test_process_input_requests_rejects_before_build_when_waiting_queue_is_full`（断言 `QueueFullError` 与消息） | ✓ 但本次实测中该路径**一次都没被走到** |
| 调度器侧"入队时拒绝" | ✓ `pipeline/test_scheduler.py::test_enqueue_built_request_honors_max_queued_requests` | ✗ 本次未观测到（两次实验均为 0 次） |
| **协调器的在飞上限（真正拦截的那一层）** | ✗ 全仓库检索 `in-flight cap` / `max_in_flight` 的断言**零命中** | ✓ 实测 3 次 |
| **`/v1/chat/completions` 队列满返回 503** | ✗ 只有语音接口有测试；**没有任何用例断言 chat 路径的 503** —— 这正是返回 500 却长期无人发现的原因 | ✗ 实测 500 |
| 两个参数"都设 / 只设一个"的语义差异 | ✗ | ✓ 实测两种语义 |
| 超限后恢复 | ✗ | ✓ |

UT 能验引擎调度器自身的拒绝逻辑，**但验不了真正生效的协调器那一层**（那需要完整管线 + 并发请求），也**没有任何用例覆盖 chat 接口的状态码**。本次实测补齐了这两块，并暴露出"拦截层级"与"状态码"与文档/直觉不符。
