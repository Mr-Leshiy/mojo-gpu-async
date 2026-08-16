# mojo-gpu-async

A single-threaded async runtime for GPU-based Mojo coroutines. It smartly
coalesces `DeviceContext.synchronize()` calls across queued tasks.

> **Warning**
> Async support in `Mojo` is experimental. APIs here may change without notice
> and should not be relied on for production use.

```mojo
from max.gpu.host import DeviceContext
from std.gpu import global_idx
from gpu_async import Context, Executor


def square_kernel(buf: Pointer[Float32, MutAnyOrigin], size: Int32):
    var idx = global_idx.x
    if idx < size:
        var value = buf[unsafe_offset=idx]
        buf[unsafe_offset=idx] = value * value


async def square[
    size: Int
](ctx: Context, input: Array[Float32, size]) raises -> Array[Float32, size]:
    var device_buffer = ctx.gpu_ctx().enqueue_create_buffer[DType.float32](size)
    ctx.gpu_ctx().enqueue_copy(
        dst_buf=device_buffer, src_ptr=input.unsafe_ptr()
    )

    ctx.gpu_ctx().enqueue_function[square_kernel](
        device_buffer, Int32(size), grid_dim=1, block_dim=size
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

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
