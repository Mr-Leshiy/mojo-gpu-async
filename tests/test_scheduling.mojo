"""Multiple tasks sharing one `Executor`: outcome correctness under
whatever interleaving the scheduler happens to pick.

Ground rule: these tests assert on outcomes only -- final results, shared
state, completion -- never on the executor's internal resume order, which
is an implementation detail, not a contract.
"""

from max.gpu.host import DeviceContext
from std.memory import OwnedPointer
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true

from warp.context import Context
from warp.executor import Executor


async def _yield_once(context: Context) -> Int:
    await context.synchronize()
    return 1


async def _yields_twice[VALUE: Int](context: Context) -> Int:
    var total = VALUE
    total += await _yield_once(context)
    total += await _yield_once(context)
    return total


def test_tasks_complete_independently() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            var first = executor.add(_yields_twice[10](context))
            var second = executor.add(_yields_twice[20](context))

            executor.wait()

            # Both ran to completion even though each parked twice on the way,
            # i.e. the queue really did round-robin between them.
            assert_true(first.is_completed())
            assert_true(second.is_completed())
            assert_equal(first^.wait(), 12)
            assert_equal(second^.wait(), 22)


async def _increment_twice(
    context: Context, counter: Pointer[Int, MutUntrackedOrigin]
):
    counter[] += 1
    await context.synchronize()
    counter[] += 1
    await context.synchronize()


def test_tasks_make_progress_without_corrupting_shared_state() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            # `counter` has to live behind a pointer, not as a bare local:
            # it's read and written from three coroutines reached through
            # `counter_ptr`'s untracked origin, a path the compiler can't see
            # aliases `counter` itself. Same issue as `_ExecutorInner._q` in
            # `warp/executor.mojo`.
            var counter = OwnedPointer(0)
            var counter_ptr = counter.ptr().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()

            var t1 = executor.add(_increment_twice(context, counter_ptr))
            var t2 = executor.add(_increment_twice(context, counter_ptr))
            var t3 = executor.add(_increment_twice(context, counter_ptr))

            executor.wait()

            assert_true(t1.is_completed())
            assert_true(t2.is_completed())
            assert_true(t3.is_completed())

            # Every increment landed exactly once: 3 tasks x 2 increments
            # each, with no lost or duplicated update despite all three being
            # interleaved on one executor. This only checks the outcome, not
            # which task's increment happened at which step -- the resume
            # order between tasks is an implementation detail, not a
            # contract (see this file's ground rule).
            assert_equal(counter_ptr[], 6)


async def _no_yield(value: Int) -> Int:
    return value


def test_mixed_yielding_and_non_yielding_tasks_complete_correctly() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            var t1 = executor.add(_no_yield(10))
            var t2 = executor.add(_yield_once(context))
            var t3 = executor.add(_no_yield(20))
            var t4 = executor.add(_yield_once(context))

            executor.wait()

            assert_equal(t1^.wait(), 10)
            assert_equal(t2^.wait(), 1)
            assert_equal(t3^.wait(), 20)
            assert_equal(t4^.wait(), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
