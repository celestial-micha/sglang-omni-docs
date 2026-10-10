# 把一个处理阶段放进独立进程：isolate_stage

## 1. 验证对象与结论

一条语音流水线可以有预处理、模型计算、声音解码等多个阶段（stage）。多个阶段可以共用一个进程，也可以分开运行。旧参数 `--isolate-stage vocoder` 的意思是把声音解码阶段分到名为 `vocoder` 的进程；它原本是 `--stage-process vocoder=vocoder` 的简写，不是选择哪张计算卡。

三类判定：③ **现行版本不需要重新支持已删除的 `isolate_stage`，Issue 参数表这一旧名称可标 X，备注“旧 CLI 已移除，改用 stage.process”。** NPU 仍可能需要进程隔离；X 只表示旧入口不再适用，不表示 NPU 不需要这项能力。

这次查到的情况与 `decode_mode` 报告一样，应先判断旧字段是否还存在。上游 [进程副本重构 #1175](https://github.com/sgl-project/sglang-omni/pull/1175) 已删除旧参数及其覆盖函数；所核对的现行源码也没有恢复。当前用 `--vocoder.process vocoder` 表达进程归属。要独立运行，该名称不能与其他非 TP 阶段共用，还须满足模型自己的拓扑限制。

本轮没有开发 NPU 专用补丁：不存在需要为现行版本补回的旧接口。已完成删除依据核对和替代字段的配置层验证，**没有把配置通过写成真实 NPU 进程启动或模型推理通过**。

## 2. 测试用例

实验道理：先核对旧参数确实被删除，再用同一份两阶段配置比较“不改设置”和“只改声音解码阶段的进程名”。前者保留共用进程，后者得到两个不同进程分组，才能说明现行配置仍能表达原来的隔离意图。这一步发生在模型加载之前。

### 2.1 环境信息

核查日期：2026-10-10。源码固定为上游提交 `cf496151dfa6eed47e7ed466e451a6e3eb1204d6`，使用独立下载副本，未改动产品代码。测试在 Windows、Python 3.12.14、Pydantic 2.14.0、pytest 9.1.1 上运行；不加载权重、不调用 NPU。配置包含 `preprocessing` 和 `vocoder`，初始均属于 `pipeline` 进程；测试调用真实配置路径解析器、解析器合并逻辑和逻辑进程编译函数，不执行占位 factory。

原始材料在工作区 `两参数验证_20261010/`：`test_process_parameters.py` 为检查入口，`evidence/config_tests.log` 为运行输出，`evidence/process_replica_commit.json` 为删除提交，`source/` 为固定源码。12 项检查是两个参数共用的总数。

### 2.2 用例测试结果

| 用例名称 | 用例步骤 | 预期结果 | 实际结果 |
| --- | --- | --- | --- |
| 旧入口是否删除 | 核对 #1175 的删除 diff，再解析现行 `serve` 函数声明 | 删除记录与现行源码一致，不再注册旧参数 | 一致；`isolate_stage` 形参及 `--isolate-stage` 注册均不存在 |
| 旧名称不能当配置使用 | 将连字符名称 `isolate-stage` 和下划线名称 `isolate_stage` 分别交给真实配置路径解析器 | 拒绝不存在的配置名称 | 两种均抛出 `ConfigPathError`；这不是完整 CLI 子进程测试 |
| 不传进程覆盖 | 对两阶段默认配置应用空覆盖，编译逻辑进程 | 保留原来的共用分组 | 两阶段仍为 `pipeline`，编译成功 |
| 使用现行隔离配置 | 仅覆盖 `vocoder.process=vocoder`，再编译逻辑进程 | 预处理保留 `pipeline`，声音解码改为 `vocoder` | 两阶段分别为 `pipeline`、`vocoder`，编译成功；原始配置未被修改 |
| 无效阶段与旧前缀 | 分别传入 `missing.process` 和 `stages.vocoder.process` | 拒绝无效阶段和不符合当前 CLI 规则的前缀 | 均被拒绝；显式前缀提示 `stages.` 已隐含，应写 `vocoder.process` |
| 缺少必需进程声明 | 构造没有 `process` 的非 TP 阶段 | 校验失败，不悄悄猜测进程 | 抛出包含 `Non-TP stages must declare process` 的错误 |

上游 [迁移指南](https://github.com/sgl-project/sglang-omni/blob/cf496151dfa6eed47e7ed466e451a6e3eb1204d6/docs/basic_usage/process_topology_migration.md) 中的替代示例仍带 `--stages.` 前缀，但现行解析器实测拒绝该写法。本报告采用 `--vocoder.process vocoder`；这是新旧文档与配置语法的差异，不能直接照抄旧示例。

## 3. UT覆盖

### 3.1 当前覆盖情况

本轮独立配置检查共 **12 passed、0 failed、0 skipped**，涵盖两个旧入口的声明检查、旧名称拒绝、默认保留、拆分与合并分组、未知阶段、旧前缀拒绝和非 TP 阶段缺少进程声明。测试只调用配置模块，未 mock NPU 成功结果。

现行仓库 `tests/unit_test/pipeline/test_topology.py` 已有 stage.process 覆盖、非 TP 进程必填、TP 进程名、同卡多进程预算及跨卡分组限制等测试。本轮阅读了这些测试，未运行整套上游 pipeline UT，不能把它们记为本轮通过数。

覆盖缺口：没有运行完整 CLI 启动测试，也没有测真实进程 PID、NPU 设备归属、跨进程数据传递或模型输出。这些属于现行进程隔离能力的运行验收，不影响“旧参数已删除，不应补回”的判定；如需验证仍保留旧参数的历史版本，应固定该版本另测，不能套用本报告的 X。
