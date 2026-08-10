# mojo-gpu-async

A single-threaded async runtime for GPU-based Mojo coroutines.

## Example

```mojo
from max.gpu.host import DeviceContext
from gpu_async import Context, Executor


async def work(ctx: Context) -> Int:
    # ... launch GPU kernels via ctx.gpu_ctx() ...
    await ctx.synchronize()  # yield to the other queued tasks
    return 42


def main() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)
        var task = executor.add(work(executor.context()))
        executor.wait()
        print(task^.wait())
```

## Development

Requires [pixi](https://pixi.sh).

```sh
pixi run fmt    # format
pixi run test   # run tests (needs a GPU-enabled host)
pixi run docs   # build the API docs site
```

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
