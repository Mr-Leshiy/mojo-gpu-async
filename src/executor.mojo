"""The queue that runs the coroutines and the device context they share."""

from max.gpu.host import DeviceContext
from std.builtin._coroutine import (
    AnyCoroutine,
    Coroutine,
    RaisingCoroutine,
    _coro_resume_fn,
    _coro_destroy_fn,
)
from std.collections import Deque
from std.memory import ArcPointer, OwnedPointer

from .context import Context
from .task import RaisingTask, Task


struct Executor(Movable):
    """Runs coroutines that share one GPU device context.

    Tasks are queued by `add` and only make progress inside `wait`, which
    resumes them in turn until every one of them has completed.
    """

    var _inner: ArcPointer[_ExecutorInner]

    def __init__(out self, ctx: DeviceContext):
        """Initialize an executor running its tasks on the given device.

        Args:
            ctx: The device context shared by every task on this executor.
        """
        self._inner = ArcPointer(_ExecutorInner(ctx))

    def context(self) -> Context:
        """Return a context that coroutines use to reach this executor."""
        return Context(self._inner.copy())

    def add[
        type: Deinitable & Movable, origins: OriginSet
    ](
        mut self,
        var handle: Coroutine[type, origins],
        out task: Task[type, origins],
    ):
        """Queue a coroutine and return the task tracking it.

        The coroutine is not started here: `wait` is what runs it.

        Args:
            handle: The coroutine to run. Ownership is transferred.
        """
        task = Task(handle^, self._inner.copy())
        self._inner[].add(task._handle, False)

    def add[
        type: Deinitable & Movable, origins: OriginSet
    ](
        mut self,
        var handle: RaisingCoroutine[type, origins],
        out task: RaisingTask[type, origins],
    ):
        """Queue a raising coroutine and return the task tracking it.

        The coroutine is not started here: `wait` is what runs it. Its error,
        if it raises one, surfaces from the returned task's `wait`.

        Args:
            handle: The raising coroutine to run. Ownership is transferred.
        """
        task = RaisingTask(handle^, self._inner.copy())
        self._inner[].add(task._handle, False)

    def wait(self) raises:
        """Run queued tasks until all have completed, then sync the device."""
        self._inner[].wait()


struct _ExecutorInner:
    """The executor state shared between the `Executor` and its `Context`s."""

    var _ctx: DeviceContext

    # The queue has to stay behind a pointer. `wait` takes `mut self` — an
    # exclusive, `noalias` borrow — yet a coroutine it resumes reaches this
    # same object through the `Context`'s executor pointer to enqueue itself.
    # Stored inline, the deque header would sit in that exclusively borrowed
    # memory and may be cached in registers across the resume, silently
    # dropping the append. Behind a pointer the header lives outside that
    # borrow and both paths agree on it. Note that the queue is genuinely
    # shared-mutable across those two paths, so `OwnedPointer`'s uniqueness
    # claim is a fiction the optimizer is free to act on. `_sync_counter`
    # below is read and written through the same two paths, for the same
    # reason, so it lives behind a pointer too.
    # (Analysis by Claude)
    var _q: OwnedPointer[Deque[AnyCoroutine]]

    # How many pops, from the front of `_q`, until we reach the first
    # coroutine that's resuming after a `Context.synchronize()` yield. `0`
    # means none is currently queued.
    #
    # A device sync is a global barrier, so firing it once, right before
    # that first tracked pop, is enough to cover every other "resuming after
    # a yield" coroutine queued behind it too — their GPU work was launched
    # even earlier in real time, so the same sync flushes it as well. That's
    # why `add` only sets this when it's `0`: anything that yields while a
    # sync is already pending rides it for free instead of scheduling a
    # redundant one.
    var _sync_counter: OwnedPointer[Int]

    def __init__(out self, ctx: DeviceContext):
        """Initialize the shared state with an empty queue.

        Args:
            ctx: The device context shared by every task on this executor.
        """
        self._ctx = ctx
        self._q = OwnedPointer(Deque[AnyCoroutine]())
        self._sync_counter = OwnedPointer(0)

    def __deinit__(deinit self):
        """Destroy every coroutine still queued."""
        try:
            while len(self._q[]) > 0:
                var handle = self._q[].popleft()
                _coro_destroy_fn(handle)
        except:
            pass

    def add(mut self, handle: AnyCoroutine, is_need_sync: Bool):
        """Queue a coroutine: freshly created, or resuming after a yield.

        Args:
            handle: The coroutine to run. The caller keeps ownership of it.
            is_need_sync: True if `handle` is resuming after
                `Context.synchronize()` suspended it, so it may depend on
                GPU work it queued right before yielding and needs the
                device synced before it runs again. False for a freshly
                created task, which hasn't launched anything yet and so
                never needs a sync of its own.
        """
        self._q[].append(handle)
        # Only the *first* pending "needs sync" coroutine claims the
        # counter — see the field comment above. Its value is `handle`'s
        # 1-indexed position in the queue once appended below (`len(_q)`
        # items already ahead of it, plus itself); `wait_until` counts pops
        # down to that exact position before resuming it.
        if is_need_sync and self._sync_counter[] == 0:
            self._sync_counter[] = len(self._q[])

    def wait(mut self) raises:
        """Run queued coroutines until all have completed."""

        @__parameter
        def never() -> Bool:
            return False

        self.wait_until[never]()

    def wait_until[predicate: def() thin capturing -> Bool](mut self) raises:
        """Run queued coroutines until `predicate` holds or the queue empties.

        Synchronizes the device lazily: once, right before resuming the
        first queued coroutine that's waiting on one — not before every
        resume, and not unconditionally after the loop.
        """

        while not predicate() and len(self._q[]) > 0:
            var handle = self._q[].popleft()

            # `handle` is the tracked "needs sync" coroutine exactly when
            # the countdown reaches 1: sync now, before resuming it — not
            # before any of the fresh/no-op coroutines popped earlier.
            if self._sync_counter[] == 1:
                self._ctx.synchronize()

            if self._sync_counter[] > 0:
                self._sync_counter[] -= 1

            _coro_resume_fn(handle)
