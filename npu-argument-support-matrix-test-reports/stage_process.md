# 指定各处理阶段归哪个进程：stage_process

## 1. 验证对象与结论

流水线的预处理、模型计算和声音解码是不同阶段（stage）。旧参数 `--stage-process preprocessing=frontend` 表示“把预处理放进 frontend 进程”；给多个阶段设置同一个进程名，就让它们共用进程。它控制的是进程归属，不是 NPU 卡号，也不等于张量并行。

三类判定：③ **现行版本不需要重新支持已删除的 `stage_process`，Issue 参数表这一旧名称可标 X，备注“旧 CLI 已移除，改用 stage.process”。** 分组部署能力对 NPU 仍有意义，不能把这一 X 写成“NPU 不需要进程分组”。

上游 [进程副本重构 #1175](https://github.com/sgl-project/sglang-omni/pull/1175) 已删除 `--stage-process`、`--isolate-stage` 及旧覆盖函数，改由 `StageConfig.process` 表达进程归属。现行命令写作 `--preprocessing.process frontend`；若另一个阶段也设为 `frontend`，两者配置上属于同一组，但仍须通过模型、设备和资源限制的校验。

本轮完成旧参数删除核对及现行分组配置检查，没有新增 NPU 专用代码。这里交付的是“旧接口是否需要继续适配”的结论，**不声称任意阶段组合已在真实 NPU 上运行通过**。

## 2. 测试用例

实验道理：旧名称报错本身不够，还应验证合法的现行写法能表达分组。用同一份两阶段配置，先保持默认，再把两者都设成 `frontend`，最后只把其中一个设成独立进程名，比较解析及编译结果。这能检查配置语义，不能代替观察操作系统中的真实进程。

### 2.1 环境信息

核查日期：2026-10-10。固定上游源码提交 `cf496151dfa6eed47e7ed466e451a6e3eb1204d6`，使用独立副本，未修改产品代码。运行环境为 Windows、Python 3.12.14、Pydantic 2.14.0、pytest 9.1.1；不加载模型或使用 NPU。配置包含 `preprocessing` 和 `vocoder`，初始都属于 `pipeline`。实际执行配置路径解析、覆盖合并和逻辑进程编译，不执行占位 factory。

复核入口为工作区 `两参数验证_20261010/test_process_parameters.py`，原始输出为 `evidence/config_tests.log`；删除提交保存在 `evidence/process_replica_commit.json`，固定源码位于 `source/`。两份报告共用这组 12 项检查，不能相加算成 24 项。

### 2.2 用例测试结果

| 用例名称 | 用例步骤 | 预期结果 | 实际结果 |
| --- | --- | --- | --- |
| 旧入口是否删除 | 对照 #1175 删除 diff 和现行 `serve` 声明 | 旧参数及其注册均不再存在 | `stage_process` 形参、`--stage-process` 注册已删除，现行源码未恢复 |
| 拒绝旧名称 | 分别把 `stage-process`、`stage_process` 交给真实配置路径解析器 | 拒绝不存在的名称，不能作为有效覆盖 | 两者均抛出 `ConfigPathError`；未声称这是完整命令行启动结果 |
| 保留默认分组 | 对默认配置应用空覆盖并编译 | 两阶段保持原分组 | 两阶段均为 `pipeline`，编译成功 |
| 合并到同一组 | 同时覆盖 `preprocessing.process=frontend`、`vocoder.process=frontend` | 两阶段归属 `frontend`，配置可编译 | 两阶段值均为 `frontend`，逻辑进程编译成功 |
| 分成不同组 | 仅覆盖 `vocoder.process=vocoder` | 预处理仍在 `pipeline`，声音解码属于另一组 | 分别得到 `pipeline`、`vocoder`，编译成功，基线配置未被修改 |
| 无效配置边界 | 传不存在的阶段名、带 `stages.` 前缀的 CLI 配置路径，以及缺少 `process` 的非 TP 阶段 | 拒绝非法写法或缺失声明 | 分别得到配置路径错误、前缀提示和非 TP 阶段必须声明进程的错误 |

当前可用写法是 `--preprocessing.process frontend`。上游 [迁移指南](https://github.com/sgl-project/sglang-omni/blob/cf496151dfa6eed47e7ed466e451a6e3eb1204d6/docs/basic_usage/process_topology_migration.md) 仍保留 `--stages.preprocessing.process frontend` 示例，但当前 CLI 配置解析器明确拒绝显式 `stages.` 前缀；本轮已用反例确认，报告采用实测有效的短写法。

## 3. UT覆盖

### 3.1 当前覆盖情况

两个参数共用的配置检查为 **12 passed、0 failed、0 skipped**。本参数相关规则包括旧名称拒绝、默认分组不变、同名合组、不同名拆组、无效阶段拒绝及非 TP 阶段必须声明 `process`。旧入口删除另有真实源码声明检查和上游删除 diff 互相印证。

仓库 `tests/unit_test/pipeline/test_topology.py` 已覆盖 dotted override、非 TP 进程必填、TP 名称派生与冲突、同卡多进程预算、单进程不得跨多卡等规则；本轮只阅读这些现有测试，未把整套上游 UT 计为已运行。

尚未覆盖完整 CLI 子进程执行、真实同进程/跨进程请求、NPU 内存占用和模型输出。因此不能用本轮 12 项检查宣称现行 `process` 能力完成 NPU 端到端验收。对于本报告的旧参数删除判定，无需恢复旧入口再做推理；仍使用旧版本的环境，应按那个版本单独分析。
