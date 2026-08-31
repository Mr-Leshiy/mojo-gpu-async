"""The handle a running coroutine has on its executor."""

from max.gpu.host import DeviceContext
from std.builtin._coroutine import (
    AnyCoroutine,
    _coro_resume_fn,
    _suspend_async,
)
from std.collections import Deque
from std.memory import ArcPointer

from .executor import _ExecutorInner


struct Context(Movable):
    """A coroutine's handle on the executor running it.

    Hands out the shared `DeviceContext` and lets the coroutine yield control
    back to the executor.
    """

    var _executor: ArcPointer[_ExecutorInner]

    def __init__(out self, var executor: ArcPointer[_ExecutorInner]):
        """Initialize a context bound to the given executor.

        Args:
            executor: The executor to yield back to. Ownership is transferred.
        """
        self._executor = executor^

    def gpu_ctx(self) -> DeviceContext:
        """Return the device context shared by every task on this executor."""
        return self._executor[]._ctx

    async def synchronize(self):
        """Suspend the calling coroutine and re-queue it on the executor.

        This is an async alternative of
        https://mojolang.org/docs/std/gpu/host/device_context/DeviceContext/#synchronize.

        Note:
            Only a coroutine spawned on the executor this context came from may
            await this. It re-queues the caller onto that executor's queue, so
            awaiting it from anywhere else hands the coroutine to a runtime that
            is not the one driving it.
        """

        def body(hdl: AnyCoroutine) {self}:
            # is_need_sync=True: `hdl` launched GPU work right before this
            # yield, so it must not resume until the device has synced.
            self._executor[].add(hdl, True)

        _suspend_async(body)


struct _CoroutineContext[P: TrivialRegisterPassable](TrivialRegisterPassable):
    """A generic completion context, assigned to a coroutine's frame.

    Replaces the stdlib's `_CoroutineContext` in the same slot, so it has to
    keep that shape: a thin callback followed by the pointer-sized payload the
    coroutine passes to it when it completes. Together the two fields must
    total 16 bytes — a thin function pointer plus one pointer-sized `P` — to
    match the size the stdlib reserves for it in the coroutine frame.

    Parameterized over `P` so callers can carry whatever pointer-sized
    payload their callback needs (e.g. a task's completion-flag pointer)
    without `_CoroutineContext` itself knowing about tasks.
    """

    comptime callback_fn_type = def(Self.P) thin -> None

    var callback: Self.callback_fn_type
    var payload: Self.P
