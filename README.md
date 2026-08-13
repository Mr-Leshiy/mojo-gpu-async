# mojo-gpu-async

A single-threaded async runtime for GPU-based Mojo coroutines. It smartly
coalesces `DeviceContext.synchronize()` calls across queued tasks.

```mojo
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from gpu_async import Context, Executor


def square_kernel(buf: Pointer[Float32, MutAnyOrigin]):
    var idx = global_idx.x
    var value = buf[unsafe_offset=idx]
    buf[unsafe_offset=idx] = value * value


async def square[
    size: Int
](ctx: Context, input: Array[Float32, size]) raises -> Array[Float32, size]:
    var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](size)
    ctx.gpu_ctx().enqueue_copy(
        dst_buf=device_buffer, src_ptr=input.unsafe_ptr()
    )
    await ctx.synchronize()  # yield to the other queued tasks

    ctx.gpu_ctx().enqueue_function[square_kernel](
        device_buffer, grid_dim=1, block_dim=size
    )

    var result = Array[Float32, size](uninitialized=True)
    ctx.gpu_ctx().enqueue_copy(
        dst_ptr=result.unsafe_ptr(), src_buf=device_buffer
    )
    await ctx.synchronize()
    return result^


def main() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var context = executor.context()

        var input1: Array[Float32, 4] = [1, 2, 3, 4]
        var t1 = executor.add(square(context, input1))

        var input2: Array[Float32, 4] = [5, 6, 7, 8]
        var t2 = executor.add(square(context, input2))

        executor.wait()
        print(t1^.wait())  # [1, 4, 9, 16]
        print(t2^.wait())  # [25, 36, 49, 64]
```

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi run fmt    # format
pixi run test   # run tests (needs a GPU-enabled host)
pixi run docs   # build the API docs site
```

## How it works

- `Executor.add()` queues a task; nothing runs until `wait()`.
- `wait()` pops queued coroutines FIFO and resumes each in turn. A
  coroutine keeps running until it returns or `await`s
  `Context.synchronize()`, which hands control back to the executor.
- Only `await ctx.synchronize()` ever needs an actual device sync, and the
  executor fires it lazily — once, right before the first coroutine that's
  waiting on one resumes. Any other coroutine queued behind it rides that
  same sync for free instead of triggering its own.

### Example

Three tasks, each yielding twice, produce the schedule traced below:

```mojo
from max.gpu.host import DeviceContext
from gpu_async import Context, Executor


async def yield_twice(ctx: Context, name: String):
    print(name, "launch 1")  # ... launch GPU work via ctx.gpu_ctx() ...
    await ctx.synchronize()

    print(name, "launch 2")  # ... launch more GPU work ...
    await ctx.synchronize()

    print(name, "done")


def main() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var t1 = executor.add(yield_twice(executor.context(), "t1"))
        var t2 = executor.add(yield_twice(executor.context(), "t2"))
        var t3 = executor.add(yield_twice(executor.context(), "t3"))

        executor.wait()

```

```mermaid
flowchart TD
    S0["t1, t2, t3"]

    S0 -->|"resume(t1) — awaits ctx.synchronize()"| S1["t2, t3, ctx.synchronize(), t1_1"]
    S1 -->|"resume(t2) — awaits ctx.synchronize();<br/>sync is already pending, so none is added"| S2["t3, ctx.synchronize(), t1_1, t2_1"]
    S2 -->|"resume(t3) — awaits ctx.synchronize();<br/>sync is already pending, so none is added"| S3["ctx.synchronize(), t1_1, t2_1, t3_1"]
    S3 -->|"the pending `ctx.synchronize()` is reached, runs once"| S4["t1_1, t2_1, t3_1"]

    S4 -->|"resume(t1_1) - awaits ctx.synchronize()"| S5["t2_1, t3_1, ctx.synchronize(), t1_2"]
    S5 -->|"resume(t2_1) - awaits ctx.synchronize();<br/>sync is already pending, so none is added"| S6["t3_1, ctx.synchronize(), t1_2, t2_2"]
    S6 -->|"resume(t3_1) - awaits ctx.synchronize();<br/>sync is already pending, so none is added"| S7["ctx.synchronize(), t1_2, t2_2, t3_2"]
    S7 -->|"the pending `ctx.synchronize()` is reached, runs once"| S8["t1_2, t2_2, t3_2"]

    S8 -->|"resume(t1_2)"| S9["t2_2, t3_2"]
    S9 -->|"resume(t2_2)"| S10["t3_2"]
    S10 -->|"resume(t3_2)"| S11["all completed"]
```

`t1_1`, `t2_1`, `t3_1` are not new tasks — each is the continuation of `t1`,
`t2`, `t3` picking up right after its `await ctx.synchronize()`. Once queued,
it's just a suspended coroutine waiting for the executor to resume it, same
as any other entry in the queue.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
