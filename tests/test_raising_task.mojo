"""`RaisingTask`: error propagation, and the `forget_deinit` bookkeeping
that keeps its unwritten result/error slot from being deinitialized.
"""

from max.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_raises

from warp.context import Context
from warp.executor import Executor


async def _raises_immediately() raises -> Int:
    raise Error("immediate failure")


async def _raises_after_yields(context: Context) raises -> Int:
    # Unlike the other coroutines in this file, this one genuinely needs a
    # real yield: the scenario under test is specifically "raises *after*
    # actually suspending and resuming", not just "raises". `synchronize()`
    # is the only suspend point `warp` exposes, so exercising it here is
    # unavoidable.
    await context.synchronize()
    await context.synchronize()
    raise Error("failure after yields")


async def _raises_inner(context: Context) raises -> Int:
    # The yield here isn't testing anything by itself -- it works around a
    # nightly-compiler bug: a nested `RaisingCoroutine` that never suspends,
    # awaited from within another raising coroutine, fails to lower to LLVM
    # ('pop.cast_from_builtin' op cannot convert to scalar dtype bool).
    # Reproduces with any parameter type, not just `Context`; a real
    # `synchronize()` here avoids it. Worth its own follow-up issue.
    await context.synchronize()
    raise Error("nested failure")


async def _raises_from_nested_coroutine(context: Context) raises -> Int:
    return await _raises_inner(context)


def test_raising_task_raises_immediately() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var task = executor.add(_raises_immediately())
            with assert_raises(contains="immediate failure"):
                _ = task^.wait()


def test_raising_task_raises_after_yields() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()
            var task = executor.add(_raises_after_yields(context))
            with assert_raises(contains="failure after yields"):
                _ = task^.wait()


def test_raising_task_raises_from_nested_coroutine() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()
            var task = executor.add(_raises_from_nested_coroutine(context))
            with assert_raises(contains="nested failure"):
                _ = task^.wait()


struct _Counter(Movable):
    """A movable handle onto a shared, heap-allocated counter cell."""

    var _inner: ArcPointer[Int]

    def __init__(out self, var inner: ArcPointer[Int]):
        self._inner = inner^


struct _DropCounter(Movable):
    """Increments a shared counter exactly once, when this value is dropped."""

    var _counter: _Counter

    def __init__(out self, var counter: _Counter):
        self._counter = counter^

    def __deinit__(deinit self):
        self._counter._inner[] += 1


async def _returns_drop_counter(counter: _Counter) raises -> _DropCounter:
    return _DropCounter(_Counter(counter._inner.copy()))


async def _raises_holding_drop_counter(
    counter: _Counter,
) raises -> _DropCounter:
    # `counter` is only here to give this coroutine the same result type
    # (`_DropCounter`) as `_returns_drop_counter` -- it never gets to
    # construct one, since it raises first.
    raise Error("boom")


def test_forget_deinit_does_not_double_drop_the_unwritten_result_slot() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)

            var counter = _Counter(ArcPointer(0))
            var task = executor.add(_raises_holding_drop_counter(counter))
            with assert_raises(contains="boom"):
                _ = task^.wait()

            # The result slot was never written -- `forget_deinit` must have
            # skipped running `_DropCounter`'s destructor over that garbage
            # memory, or this would already have corrupted/crashed.
            assert_equal(counter._inner[], 0)


def test_forget_deinit_does_not_leak_a_written_result_on_the_success_path() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)

            var counter = _Counter(ArcPointer(0))
            var task = executor.add(_returns_drop_counter(counter))
            var result = task^.wait()
            # Not dropped yet: it's still held in `result`.
            assert_equal(counter._inner[], 0)
            _ = result^
            assert_equal(counter._inner[], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
