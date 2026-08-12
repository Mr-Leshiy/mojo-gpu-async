"""Tasks: the queued coroutines, their results, and their completion flags."""

from std.atomic import Atomic
from std.builtin.coroutine import AnyCoroutine
from std.memory import ArcPointer

from .context import _CoroutineContext
from .executor import _ExecutorInner


struct Task[type: Deinitable & Movable, origins: OriginSet](
    Movable where False
):
    """A coroutine queued on an `Executor`, and the result it will produce.

    Immovable: the coroutine writes its result and completion flag through
    pointers into this struct.
    """

    comptime _COMPLETED_FLAG_TYPE = DType.uint8
    """Flag type of the completion flag: `Atomic` cannot store a `Bool`'s `i1`."""

    comptime _CompletedFlagPointer = Pointer[
        Atomic[Self._COMPLETED_FLAG_TYPE], MutUntrackedOrigin
    ]
    """Pointer to a task's completion flag, as the coroutine frame holds it."""

    var _executor: ArcPointer[_ExecutorInner]
    var _handle: AnyCoroutine
    var _completed: Atomic[Self._COMPLETED_FLAG_TYPE]
    var _result: Self.type

    def __init__(
        out self,
        var handle: Coroutine[Self.type, Self.origins],
        var executor: ArcPointer[_ExecutorInner],
    ):
        """Initialize a task with a coroutine.

        Takes ownership of the provided coroutine and points it at this task's
        result slot and completion flag.

        Args:
            handle: The coroutine to execute as a task. Ownership is
                transferred.
            executor: The executor running the coroutine. Ownership is
                transferred.
        """
        self._executor = executor^
        self._completed = Atomic[Self._COMPLETED_FLAG_TYPE](0)

        __mlir_op.`lit.ownership.mark_initialized`(
            __get_mvalue_as_litref(self._result)
        )
        handle._set_result_slot(Pointer(to=self._result))

        _install_completion_callback[Self.type, Self.origins](
            handle,
            Self._CompletedFlagPointer(
                unsafe_from_address=Int(Pointer(to=self._completed))
            ),
        )

        self._handle = handle^._take_handle()

    def wait(deinit self) raises -> Self.type:
        """Run the executor until this task completes, then take its result.

        Consumes the task: the flag and the result slot it owns die with it.
        """

        @parameter
        def completed() -> Bool:
            return self.is_completed()

        self._executor[].wait_until[completed]()
        return self._result^

    def is_completed(self) -> Bool:
        """Return True once the coroutine has run to completion.

        A task that has not started, or that is parked on an `await`, reads as
        False; once True, the result is there.
        """
        return self._completed.load() != 0


def _install_completion_callback[
    type: Deinitable & Movable, origins: OriginSet
](
    mut handle: Coroutine[type, origins],
    completed: Task[type, origins]._CompletedFlagPointer,
):
    """Install the completion callback in a task coroutine's frame.

    Args:
        handle: The coroutine to install the callback on.
        completed: The flag to raise once the coroutine completes.
    """

    def _mark_completed(flag: Task[type, origins]._CompletedFlagPointer):
        flag[].store(1)

    var ctx = handle._get_ctx[
        _CoroutineContext[Task[type, origins]._CompletedFlagPointer]
    ]()
    ctx[].callback = _mark_completed
    ctx[].payload = completed
