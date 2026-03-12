# ToolOceansInfra

`tooloceans` 是一个面向工具调用轨迹采集与回放的轻量基础设施库。它把一次 agent / workflow 执行拆成一组稳定的数据结构，并用一组最小协议把工具注册、执行、事件分发、热存储、冷存储、后处理连接起来。

这个仓库当前更接近“基础协议层”和“参考实现”，适合用来做下面几类事情：

- 记录 agent 的工具调用轨迹，形成可训练、可回放的 episode 数据
- 在执行过程中实时分发事件，接入监控、UI、调试器或在线缓存
- 在执行结束后持久化完整 episode，供离线分析、评估或再训练
- 在不改上层数据结构的前提下，替换执行器、事件总线、存储后端

## 安装

```bash
pip install -e .
```

要求：

- Python `>=3.11`

## 核心设计

库的核心由两部分组成：

- 稳定的数据结构：定义一次运行中“发生了什么”
- 可替换的协议接口：定义这些数据如何流动

整体关系可以概括为：

```text
RunContext
   |
   v
EpisodeManager
   |- 调用 Executor.execute(ToolCall, RunContext)
   |- 向 EventBus.emit(Event) 发事件
   |- 最终把 Episode 保存到 ColdStore
   `- 可选执行 PostEpisodeHook

ToolRegistry
   `- 为 Executor 提供 ToolSpec + ToolHandler

HotStore
   `- 通常作为 EventBus 的订阅者实时缓存 Event
```

## 数据结构

### 1. RunContext

定义位置：[tooloceans/context.py](/Users/apple/ToolOceansInfra/tooloceans/context.py)

`RunContext` 是一次运行的上下文对象，贯穿 episode 生命周期。

字段：

- `run_id: str`
  用于标识一次完整运行。默认自动生成 UUID。
- `episode_id: str`
  用于标识当前 episode。默认自动生成 UUID。
- `step_id: str | None`
  当前 step 的标识。进入某个 step 时由 `ctx.step(step_id)` 派生。
- `trace_id: str | None`
  用于与外部 tracing 系统关联。
- `timeout_seconds: float | None`
  本次运行显式指定的超时时间。若存在，会优先于 tool capability 中的超时提示。
- `experiment_metadata: dict[str, Any]`
  实验或任务级元信息。

语义说明：

- `RunContext` 是“运行态上下文”，不是持久化轨迹本身。
- `step()` 会返回一个新的上下文副本，不会原地修改原对象。

### 2. ToolCapability

定义位置：[tooloceans/registry.py](/Users/apple/ToolOceansInfra/tooloceans/registry.py)

`ToolCapability` 描述工具的行为属性，用于调度、测试、回放或未来的执行优化。

字段：

- `has_side_effects`
  工具是否有外部副作用，默认 `True`
- `is_deterministic`
  相同输入是否稳定产生相同输出，默认 `False`
- `is_idempotent`
  重复执行是否安全，默认 `False`
- `is_pure_read`
  是否只读，默认 `False`
- `supports_mock`
  是否支持 mock 执行，默认 `False`
- `timeout_hint_seconds`
  工具建议超时时间
- `max_concurrency`
  建议最大并发度

当前实现说明：

- `timeout_hint_seconds` 会被默认执行器使用
- `max_concurrency` 当前只是元数据，默认实现没有执行并发控制

### 3. ToolSpec

定义位置：[tooloceans/registry.py](/Users/apple/ToolOceansInfra/tooloceans/registry.py)

`ToolSpec` 是工具的静态描述。

字段：

- `name: str`
- `version: str`
- `input_schema: dict[str, Any]`
- `output_schema: dict[str, Any]`
- `capability: ToolCapability`

语义说明：

- `name + version` 共同标识一个具体工具实现
- `input_schema` 和 `output_schema` 当前只做存储与暴露，不做内建校验

### 4. ToolCall

定义位置：[tooloceans/trajectory.py](/Users/apple/ToolOceansInfra/tooloceans/trajectory.py)

表示一次工具调用请求。

字段：

- `tool_name: str`
- `arguments: dict[str, Any]`
- `call_id: str`

语义说明：

- `tool_name` 只包含工具名，不包含版本
- 默认执行器会通过 registry 查找该名称的“最新版本”实现
- `call_id` 用于把请求、结果和事件关联起来

### 5. ToolError

定义位置：[tooloceans/trajectory.py](/Users/apple/ToolOceansInfra/tooloceans/trajectory.py)

表示一次工具调用失败。

字段：

- `code: str`
- `message: str`
- `detail: dict[str, Any]`
- `retryable: bool`

默认执行器当前会生成两类错误：

- `timeout`
- `execution_error`

### 6. ToolResult

定义位置：[tooloceans/trajectory.py](/Users/apple/ToolOceansInfra/tooloceans/trajectory.py)

表示一次工具调用的结果。

字段：

- `call_id: str`
- `output: Any`
- `error: ToolError | None`
- `duration_ms: float | None`

语义说明：

- 成功时 `error is None`
- 失败时 `output` 通常为 `None`，错误信息写入 `error`

### 7. Step

定义位置：[tooloceans/trajectory.py](/Users/apple/ToolOceansInfra/tooloceans/trajectory.py)

`Step` 表示 episode 中的一个离散步骤。

字段：

- `step_id: str`
- `tool_calls: list[ToolCall]`
- `tool_results: list[ToolResult]`
- `observation: Any`
- `reward: float | None`

语义说明：

- 一个 step 可以包含多个 `ToolCall`
- 当前 `EpisodeManager` 会把同一 step 的结果按顺序聚合为一个 `observation`
- `reward` 是预留字段，默认流程不会自动填充

当前默认 observation 规则：

- 成功结果写入原始 `output`
- 失败结果写入 `{"error": error.code}`

### 8. Episode

定义位置：[tooloceans/trajectory.py](/Users/apple/ToolOceansInfra/tooloceans/trajectory.py)

`Episode` 是完整轨迹的持久化单元。

字段：

- `episode_id: str`
- `run_id: str`
- `steps: list[Step]`
- `metadata: dict[str, Any]`
- `terminal_reward: float | None`

辅助方法：

- `as_rl_trajectory() -> list[dict]`

返回格式：

```python
[
    {
        "action": step.tool_calls,
        "observation": step.observation,
        "reward": step.reward,
    },
    ...
]
```

这说明 `Episode` 已经天然接近 RL / trajectory learning 常见的数据组织方式。

### Dataset Builder 选择

- canonical RL transition schema:

```python
{
    "run_id": str | None,
    "episode_id": str,
    "step_id": str | None,
    "trace_id": str | None,
    "observation": Any,
    "actions": [{"tool_name": str, "arguments": dict, "call_id": str}],
    "reward": float | None,
    "next_observation": Any,
    "done": bool,
}
```

- `EpisodeDatasetBuilder`
  兼容旧的 episode-shaped 输出；适合已经依赖 `episode -> steps[]` 结构的离线分析或旧代码路径。
- `OfflineTransitionDatasetBuilder`
  当前推荐的 offline RL 主入口；直接输出 canonical `transition[]`。
- `OnlineTransitionDatasetBuilder`
  当前推荐的 online RL 主入口；从事件流增量构建同一套 canonical `transition[]`。

在线 builder 语义：

- `build()`
  只返回已经被后台 worker 处理完成的 transition 快照，不会隐式 flush 队列。
- `drain()`
  等待当前队列中的事件全部处理完成；在读取完整在线样本前应先调用。
- `aclose()`
  先 drain，再停止当前 worker；后续如果再次收到事件，会懒启动新的 worker。

offline `trace_id` 策略：

- `OfflineTransitionDatasetBuilder` 中的 `trace_id` 固定为 `None`
  因为当前 `Episode` 持久化结构不包含 `trace_id`。

## 协议层

### ToolHandler 协议

定义位置：[tooloceans/registry.py](/Users/apple/ToolOceansInfra/tooloceans/registry.py)

```python
async def __call__(arguments: dict[str, Any], ctx: RunContext) -> Any
```

要求：

- 必须是异步可调用对象
- 输入是结构化参数和运行上下文
- 返回值可以是任意 Python 对象，默认实现不会强制校验 `output_schema`

### ToolRegistry 协议

定义位置：[tooloceans/registry.py](/Users/apple/ToolOceansInfra/tooloceans/registry.py)

接口：

```python
def register(self, spec: ToolSpec, handler: ToolHandler) -> None
def get(self, name: str, version: str | None = None) -> tuple[ToolSpec, ToolHandler]
def list(self) -> list[ToolSpec]
```

默认实现：[tooloceans/impl/registry.py](/Users/apple/ToolOceansInfra/tooloceans/impl/registry.py)

`InMemoryToolRegistry` 的行为：

- 内部以 `(name, version)` 为 key 保存工具
- `get(name, version)` 时会精确匹配
- `get(name)` 时会返回该名字下“按字符串排序后的最大版本”

重要约束：

- 这里的“最新版本”是字符串排序，不是语义化版本解析
- 例如 `"10.0"` 会排在 `"2.0"` 之后是对的，但 `"2.10"` 与 `"2.9"` 的排序依赖字符串规则，不一定符合 semver 预期

### Executor 协议

定义位置：[tooloceans/executor.py](/Users/apple/ToolOceansInfra/tooloceans/executor.py)

```python
async def execute(self, call: ToolCall, ctx: RunContext) -> ToolResult
```

默认实现：[tooloceans/impl/executor.py](/Users/apple/ToolOceansInfra/tooloceans/impl/executor.py)

`AsyncExecutor` 的执行流程：

1. 发出 `tool.requested` 事件
2. 从 `ToolRegistry` 获取工具定义和处理器
3. 计算超时：
   `ctx.timeout_seconds` 优先，否则使用 `spec.capability.timeout_hint_seconds`
4. 调用 handler
5. 成功时返回 `ToolResult(output=..., duration_ms=...)` 并发出 `tool.completed`
6. 超时时返回 `ToolResult(error=timeout)` 并发出 `tool.failed`
7. 其他异常返回 `ToolResult(error=execution_error)` 并发出 `tool.failed`

错误语义：

- `timeout` 的 `retryable=True`
- `execution_error` 的 `retryable=False`

### EventBus / EventHandler 协议

定义位置：

- [tooloceans/bus.py](/Users/apple/ToolOceansInfra/tooloceans/bus.py)
- [tooloceans/events.py](/Users/apple/ToolOceansInfra/tooloceans/events.py)

接口：

```python
class EventHandler(Protocol):
    async def handle(self, event: Event) -> None: ...

class EventBus(Protocol):
    async def emit(self, event: Event) -> None: ...
    async def subscribe(self, handler: EventHandler) -> None: ...
```

事件类型 `EventType`：

- `episode.started`
- `episode.ended`
- `step.started`
- `step.ended`
- `tool.requested`
- `tool.completed`
- `tool.failed`
- `observation.emitted`
- `reward.attached`

`Event` 字段：

- `type: EventType`
- `run_id: str`
- `episode_id: str`
- `event_id: str`
- `step_id: str | None`
- `timestamp: float`
- `payload: dict[str, Any]`
- `trace_id: str | None`

默认实现：[tooloceans/impl/bus.py](/Users/apple/ToolOceansInfra/tooloceans/impl/bus.py)

`InProcessEventBus` 的行为：

- 订阅者保存在内存列表中
- `emit()` 时按订阅顺序串行调用所有 handler
- 某个 handler 很慢时，会直接阻塞后续 handler 和上层执行流

这意味着当前事件总线是“进程内、同步等待、串行分发”的参考实现，不是高吞吐异步消息系统。

### HotStore / ColdStore 协议

定义位置：[tooloceans/storage.py](/Users/apple/ToolOceansInfra/tooloceans/storage.py)

接口：

```python
class HotStore(Protocol):
    async def emit(self, event: Event) -> None: ...

class ColdStore(Protocol):
    async def save_episode(self, episode: Episode) -> None: ...
    async def load_episode(self, episode_id: str) -> Episode | None: ...
```

#### HotStore

默认实现：[tooloceans/impl/hot_store.py](/Users/apple/ToolOceansInfra/tooloceans/impl/hot_store.py)

`InMemoryHotStore` 同时实现了：

- `handle(event)`，便于作为 `EventBus` 订阅者使用
- `emit(event)`，便于直接以 hot store 形式写入

附加能力：

- `events()` 返回事件副本列表
- `events_for_episode(episode_id)` 过滤某个 episode 的事件

适用场景：

- 调试
- 单元测试
- 简单 UI 的实时事件缓存

#### ColdStore

默认实现：[tooloceans/impl/cold_store.py](/Users/apple/ToolOceansInfra/tooloceans/impl/cold_store.py)

`LocalFileColdStore` 的行为：

- 保存路径为 `<base_path>/<episode_id>.jsonl`
- 当前每个文件只写入 1 行 JSON
- `load_episode()` 会从该 JSON 重建 `Episode`

注意：

- 文件扩展名是 `.jsonl`，但当前实现不是“多行逐条追加”的流式日志，而是单行完整 episode 快照
- 如果未来要做增量写入或多对象日志，需要显式升级格式约定

### PostEpisodeHook 协议

定义位置：[tooloceans/hooks.py](/Users/apple/ToolOceansInfra/tooloceans/hooks.py)

```python
async def on_episode_end(self, episode: Episode) -> Episode
```

用途：

- 在持久化前补充 reward
- 打标签
- 清洗或压缩 metadata
- 导出额外索引

`EpisodeManager` 会在 `episode.ended` 事件之后、`save_episode()` 之前依次执行所有 hook。

## EpisodeManager 执行协议

定义位置：[tooloceans/episode_manager.py](/Users/apple/ToolOceansInfra/tooloceans/episode_manager.py)

`EpisodeManager.run_episode()` 是整个系统的编排入口：

```python
async def run_episode(
    ctx: RunContext,
    steps_input: list[list[ToolCall]],
    metadata: dict[str, Any] | None = None,
) -> Episode
```

其中：

- `steps_input` 的外层 list 表示 step 序列
- 每个 step 内层 list 表示该 step 的全部工具调用

当前默认流程：

1. 创建 `Episode`
2. 发 `episode.started`
3. 对每个 step：
   - 生成新的 `step_id`
   - 发 `step.started`
   - 逐个执行该 step 内的 `ToolCall`
   - 聚合 `observation`
   - 发 `observation.emitted`
   - 把 `Step` 追加到 `Episode.steps`
   - 发 `step.ended`
4. 发 `episode.ended`
5. 依次运行 `PostEpisodeHook`
6. 调用 `ColdStore.save_episode`
7. 返回完整 `Episode`

重要实现细节：

- `steps_input` 是“按 step 分组”的输入协议，不等于“step 内一定并发执行”
- 当前实现里，同一 step 内的多个 tool call 是按顺序 `await` 执行的
- 如果想真正并发，需要替换 `EpisodeManager` 或 `Executor` 编排逻辑

## 事件协议细节

下面是默认实现下事件的大致时序：

```text
episode.started
step.started
tool.requested
tool.completed | tool.failed
...（该 step 中每个调用重复）
observation.emitted
step.ended
...（每个 step 重复）
episode.ended
```

典型 payload：

### `tool.requested`

```json
{
  "tool_name": "add",
  "call_id": "c1"
}
```

### `tool.completed`

```json
{
  "call_id": "c1",
  "duration_ms": 3.42
}
```

### `tool.failed`

```json
{
  "call_id": "c1",
  "error_code": "timeout"
}
```

### `observation.emitted`

```json
{
  "observation": [
    {"result": 3},
    {"error": "timeout"}
  ]
}
```

说明：

- 事件里保留了 `run_id / episode_id / step_id / trace_id`，便于串起完整链路
- `reward.attached` 已被定义为事件类型，但默认 `EpisodeManager` 当前不会主动发该事件

## 持久化格式

`LocalFileColdStore` 写出的文件是单行 JSON。逻辑文件名示例：

```text
/tmp/tooloceans_episodes/<episode_id>.jsonl
```

示例内容：

```json
{
  "episode_id": "ep-001",
  "run_id": "run-001",
  "steps": [
    {
      "step_id": "step-001",
      "tool_calls": [
        {
          "tool_name": "add",
          "arguments": {"a": 1, "b": 2},
          "call_id": "call-001"
        }
      ],
      "tool_results": [
        {
          "call_id": "call-001",
          "output": {"result": 3},
          "error": null,
          "duration_ms": 1.2
        }
      ],
      "observation": [
        {"result": 3}
      ],
      "reward": null
    }
  ],
  "metadata": {
    "task": "demo"
  },
  "terminal_reward": null
}
```

反序列化时会重建：

- `Episode`
- `Step`
- `ToolCall`
- `ToolResult`
- `ToolError`

当前边界：

- 假定文件内容是由本库写入的合法结构
- 不做严格 schema 校验
- 不处理跨版本迁移

## 端到端示例

示例文件：[examples/basic_run.py](/Users/apple/ToolOceansInfra/examples/basic_run.py)

最小使用方式：

```python
import asyncio
import uuid
from tooloceans.context import RunContext
from tooloceans.trajectory import ToolCall
from tooloceans.registry import ToolSpec, ToolCapability
from tooloceans.episode_manager import EpisodeManager
from tooloceans.impl.registry import InMemoryToolRegistry
from tooloceans.impl.executor import AsyncExecutor
from tooloceans.impl.bus import InProcessEventBus
from tooloceans.impl.hot_store import InMemoryHotStore
from tooloceans.impl.cold_store import LocalFileColdStore


async def add(arguments: dict, ctx: RunContext) -> dict:
    return {"result": arguments["a"] + arguments["b"]}


async def main() -> None:
    registry = InMemoryToolRegistry()
    registry.register(
        ToolSpec(
            name="add",
            version="1.0",
            input_schema={"a": "number", "b": "number"},
            output_schema={"result": "number"},
            capability=ToolCapability(
                has_side_effects=False,
                is_deterministic=True,
                is_pure_read=True,
            ),
        ),
        add,
    )

    bus = InProcessEventBus()
    hot_store = InMemoryHotStore()
    await bus.subscribe(hot_store)

    cold_store = LocalFileColdStore("/tmp/tooloceans_episodes")
    executor = AsyncExecutor(registry, bus)
    manager = EpisodeManager(executor, bus, cold_store)

    ctx = RunContext(experiment_metadata={"experiment": "basic_run"})
    steps_input = [[
        ToolCall(
            tool_name="add",
            arguments={"a": 1, "b": 2},
            call_id=str(uuid.uuid4()),
        )
    ]]

    episode = await manager.run_episode(ctx, steps_input, metadata={"task": "demo"})
    print(episode.as_rl_trajectory())


asyncio.run(main())
```

## 默认实现的能力边界

当前仓库已经覆盖了最小闭环，但它仍然是偏基础设施原型的实现。文档上需要明确以下边界：

- `input_schema` / `output_schema` 仅保存，不自动校验
- step 内调用当前是顺序执行，不是并发执行
- 事件总线是串行进程内分发
- cold store 当前是单文件单行 JSON 快照
- reward 相关字段和事件已预留，但默认流程不自动生成
- registry 的“最新版本”选择基于字符串排序

这套设计的优点是接口小、改造成本低，适合作为上层 agent infra 的内核骨架。

## 包导出

`tooloceans` 在顶层导出了最重要的协议和数据结构，见 [tooloceans/__init__.py](/Users/apple/ToolOceansInfra/tooloceans/__init__.py)：

- `RunContext`
- `Event`, `EventType`
- `Episode`, `Step`, `ToolCall`, `ToolResult`, `ToolError`
- `ToolSpec`, `ToolCapability`, `ToolRegistry`, `ToolHandler`
- `Executor`
- `EventBus`, `EventHandler`
- `HotStore`, `ColdStore`
- `PostEpisodeHook`

## 测试

运行：

```bash
pytest
```

当前测试覆盖了：

- registry 注册与版本选择
- executor 成功、异常、超时与事件发出
- event bus 订阅与分发
- hot store 缓存行为
- cold store 的保存与回读
- episode manager 的端到端执行
