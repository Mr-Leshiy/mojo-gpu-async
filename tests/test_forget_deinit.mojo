"""`forget_deinit` bookkeeping: a task's unwritten result/error slot must
not be deinitialized, and a written result must be dropped exactly once.
"""

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_raises

from warp.executor import Executor


@fieldwise_init
struct _Counter(Movable):
    """A movable handle onto a shared, heap-allocated counter cell."""

    var _inner: ArcPointer[Int]


@fieldwise_init
struct _DropCounter(Movable):
    """Increments a shared counter exactly once, when this value is dropped."""

    var _counter: _Counter

    def __deinit__(deinit self):
        self._counter._inner[] += 1


async def _constructs_drop_counter_then_maybe_raises(
    counter: _Counter, should_raise: Bool
) raises -> _DropCounter:
    var drop_counter = _DropCounter(_Counter(counter._inner.copy()))
    if should_raise:
        raise Error("boom")
    return drop_counter^


def test_forget_deinit_does_not_double_drop_the_unwritten_result_slot() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)

            var counter = _Counter(ArcPointer(0))
            var task = executor.add(
                _constructs_drop_counter_then_maybe_raises(counter, True)
            )
            with assert_raises(contains="boom"):
                _ = task^.wait()

            # The result slot itself was never written -- only the local
            # `drop_counter` was dropped once, when the coroutine unwound.
            # If `forget_deinit` also ran `_DropCounter`'s destructor over
            # that unwritten slot, this would read 2 (or have already
            # corrupted/crashed).
            assert_equal(counter._inner[], 1)


def test_forget_deinit_does_not_leak_a_written_result_on_the_success_path() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)

            var counter = _Counter(ArcPointer(0))
            var task = executor.add(
                _constructs_drop_counter_then_maybe_raises(counter, False)
            )
            var result = task^.wait()
            # Not dropped yet: it's still held in `result`.
            assert_equal(counter._inner[], 0)
            _ = result^
            assert_equal(counter._inner[], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
