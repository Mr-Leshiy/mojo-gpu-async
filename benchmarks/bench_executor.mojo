"""Benchmark: `Executor`'s sync-coalescing scheduling against running the
same tasks one at a time, and against not using an executor at all.

Each task is a tiny two-stage GPU pipeline (copy in, square, copy out).

- "regular" doesn't touch `Context`/`Executor`/coroutines at all: it drives
  `DeviceContext` directly, one pipeline fully finished before the next
  starts. The baseline the other two scenarios are measured against.
- "sync" runs the same pipelines through the executor, but adds one task
  and waits on it before adding the next, so every yield pays for its own
  device sync — no overlap, just routed through `Context.synchronize()`
  instead of a direct blocking call.
- "async" queues every task up front and waits once, so `Executor`
  coalesces all of a generation's yields into a single sync no matter how
  many tasks are in flight.

Run with:
    mojo run -O3 -I . benchmarks/bench_executor.mojo
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.gpu import global_idx
from std.sys import has_accelerator

from gpu_async.context import Context
from gpu_async.executor import Executor

comptime SIZE = 256
"""Elements per task; also the kernel's block dimension."""


def square_kernel(buf: Pointer[Float32, MutAnyOrigin]):
    var idx = global_idx.x

    var value = buf[unsafe_offset=idx]
    buf[unsafe_offset=idx] = value * value


async def square(
    ctx: Context, input: Array[Float32, SIZE]
) raises -> Array[Float32, SIZE]:
    var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](
        SIZE
    )
    ctx.gpu_ctx().enqueue_copy(
        dst_buf=device_buffer, src_ptr=input.unsafe_ptr()
    )
    await ctx.synchronize()

    ctx.gpu_ctx().enqueue_function[square_kernel](
        device_buffer, grid_dim=1, block_dim=SIZE
    )

    var result = Array[Float32, SIZE](uninitialized=True)
    ctx.gpu_ctx().enqueue_copy(
        dst_ptr=result.unsafe_ptr(), src_buf=device_buffer
    )
    await ctx.synchronize()
    return result^


def square_regular(
    ctx: DeviceContext, input: Array[Float32, SIZE]
) raises -> Array[Float32, SIZE]:
    """The same pipeline as `square`, but plain and blocking.

    No `Context`, no `Executor`, no coroutine: every stage runs to
    completion, synced, before the next one starts.
    """
    var device_buffer = ctx.enqueue_create_buffer[DType.float32](SIZE)
    ctx.enqueue_copy(dst_buf=device_buffer, src_ptr=input.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[square_kernel](
        device_buffer, grid_dim=1, block_dim=SIZE
    )

    var result = Array[Float32, SIZE](uninitialized=True)
    ctx.enqueue_copy(dst_ptr=result.unsafe_ptr(), src_buf=device_buffer)
    ctx.synchronize()
    return result^


def make_input(seed: Int) -> Array[Float32, SIZE]:
    var arr = Array[Float32, SIZE](uninitialized=True)
    for i in range(SIZE):
        arr[i] = Float32(seed + i)
    return arr^


def bench_regular_4(ctx: DeviceContext) raises:
    """Run 4 pipelines back-to-back with plain, blocking `DeviceContext`
    calls — no `Context`/`Executor`/coroutines involved.
    """

    @parameter
    def body() raises:
        for i in range(4):
            var result = square_regular(ctx, make_input(i))
            keep(result.unsafe_ptr())

    run[body]().print("regular_4tasks")


def bench_regular_8(ctx: DeviceContext) raises:
    """Run 8 pipelines back-to-back with plain, blocking `DeviceContext`
    calls — no `Context`/`Executor`/coroutines involved.
    """

    @parameter
    def body() raises:
        for i in range(8):
            var result = square_regular(ctx, make_input(i))
            keep(result.unsafe_ptr())

    run[body]().print("regular_8tasks")


def bench_sync_4(ctx: DeviceContext) raises:
    """Run 4 pipelines one at a time: add, then wait immediately.

    No task overlaps another, so every yield pays for its own device sync.
    """

    @parameter
    def body() raises:
        var executor = Executor(ctx)

        var t0 = executor.add(square(executor.context(), make_input(0)))
        var r0 = t0^.wait()
        keep(r0.unsafe_ptr())
        var t1 = executor.add(square(executor.context(), make_input(1)))
        var r1 = t1^.wait()
        keep(r1.unsafe_ptr())
        var t2 = executor.add(square(executor.context(), make_input(2)))
        var r2 = t2^.wait()
        keep(r2.unsafe_ptr())
        var t3 = executor.add(square(executor.context(), make_input(3)))
        var r3 = t3^.wait()
        keep(r3.unsafe_ptr())

    run[body]().print("sync_4tasks")


def bench_async_4(ctx: DeviceContext) raises:
    """Queue 4 pipelines up front, then wait once.

    Every task's first stage launches before the executor's lazy sync
    fires, so all 4 share it instead of each triggering their own.
    """

    @parameter
    def body() raises:
        var executor = Executor(ctx)

        var t0 = executor.add(square(executor.context(), make_input(0)))
        var t1 = executor.add(square(executor.context(), make_input(1)))
        var t2 = executor.add(square(executor.context(), make_input(2)))
        var t3 = executor.add(square(executor.context(), make_input(3)))

        executor.wait()

        var r0 = t0^.wait()
        keep(r0.unsafe_ptr())
        var r1 = t1^.wait()
        keep(r1.unsafe_ptr())
        var r2 = t2^.wait()
        keep(r2.unsafe_ptr())
        var r3 = t3^.wait()
        keep(r3.unsafe_ptr())

    run[body]().print("async_4tasks")


def bench_sync_8(ctx: DeviceContext) raises:
    """Run 8 pipelines one at a time: add, then wait immediately."""

    @parameter
    def body() raises:
        var executor = Executor(ctx)

        var t0 = executor.add(square(executor.context(), make_input(0)))
        var r0 = t0^.wait()
        keep(r0.unsafe_ptr())
        var t1 = executor.add(square(executor.context(), make_input(1)))
        var r1 = t1^.wait()
        keep(r1.unsafe_ptr())
        var t2 = executor.add(square(executor.context(), make_input(2)))
        var r2 = t2^.wait()
        keep(r2.unsafe_ptr())
        var t3 = executor.add(square(executor.context(), make_input(3)))
        var r3 = t3^.wait()
        keep(r3.unsafe_ptr())
        var t4 = executor.add(square(executor.context(), make_input(4)))
        var r4 = t4^.wait()
        keep(r4.unsafe_ptr())
        var t5 = executor.add(square(executor.context(), make_input(5)))
        var r5 = t5^.wait()
        keep(r5.unsafe_ptr())
        var t6 = executor.add(square(executor.context(), make_input(6)))
        var r6 = t6^.wait()
        keep(r6.unsafe_ptr())
        var t7 = executor.add(square(executor.context(), make_input(7)))
        var r7 = t7^.wait()
        keep(r7.unsafe_ptr())

    run[body]().print("sync_8tasks")


def bench_async_8(ctx: DeviceContext) raises:
    """Queue 8 pipelines up front, then wait once."""

    @parameter
    def body() raises:
        var executor = Executor(ctx)

        var t0 = executor.add(square(executor.context(), make_input(0)))
        var t1 = executor.add(square(executor.context(), make_input(1)))
        var t2 = executor.add(square(executor.context(), make_input(2)))
        var t3 = executor.add(square(executor.context(), make_input(3)))
        var t4 = executor.add(square(executor.context(), make_input(4)))
        var t5 = executor.add(square(executor.context(), make_input(5)))
        var t6 = executor.add(square(executor.context(), make_input(6)))
        var t7 = executor.add(square(executor.context(), make_input(7)))

        executor.wait()

        var r0 = t0^.wait()
        keep(r0.unsafe_ptr())
        var r1 = t1^.wait()
        keep(r1.unsafe_ptr())
        var r2 = t2^.wait()
        keep(r2.unsafe_ptr())
        var r3 = t3^.wait()
        keep(r3.unsafe_ptr())
        var r4 = t4^.wait()
        keep(r4.unsafe_ptr())
        var r5 = t5^.wait()
        keep(r5.unsafe_ptr())
        var r6 = t6^.wait()
        keep(r6.unsafe_ptr())
        var r7 = t7^.wait()
        keep(r7.unsafe_ptr())

    run[body]().print("async_8tasks")


def main() raises:
    comptime if not has_accelerator():
        print("No accelerator available; skipping benchmark.")
        return

    print("Running Executor scheduling benchmarks")

    with DeviceContext() as ctx:
        bench_regular_4(ctx)
        bench_regular_8(ctx)
        bench_sync_4(ctx)
        bench_async_4(ctx)
        bench_sync_8(ctx)
        bench_async_8(ctx)
