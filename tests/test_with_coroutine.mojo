from max.gpu.host import DeviceContext
from std.memory import OwnedPointer
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from gpu_async.context import Context
from gpu_async.executor import Executor


async def _yield_once(context: Context) -> Int:
    await context.synchronize()
    return 1


async def _yields_twice[VALUE: Int](context: Context) -> Int:
    var total = VALUE
    total += await _yield_once(context)
    total += await _yield_once(context)
    return total


def test_task_completes_with_its_result() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            var task = executor.add(_yields_twice[5](context))
            assert_false(task.is_completed())

            executor.wait()

            assert_true(task.is_completed())
            assert_equal(task^.wait(), 7)


def test_task_wait_returns_its_result() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            var first = executor.add(_yields_twice[5](context))
            var second = executor.add(_yields_twice[10](context))

            assert_equal(first^.wait(), 7)
            assert_false(second.is_completed())
            assert_equal(second^.wait(), 12)


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


async def _record_step(
    context: Context, step: Pointer[Int, MutUntrackedOrigin]
) -> Int:
    var s0 = step[]
    step[] += 1
    await context.synchronize()

    var s1 = step[]
    step[] += 1
    await context.synchronize()

    var s2 = step[]
    step[] += 1

    return s0 * 100 + s1 * 10 + s2


def test_tasks_resume_round_robin_not_one_at_a_time() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            # `step` has to live behind a pointer, not as a bare local: it's
            # read and written from three coroutines reached through
            # `step_ptr`'s untracked origin, a path the compiler can't see
            # aliases `step` itself. A bare stack slot may be kept in a
            # register at each of those disconnected access points instead of
            # actually written back, so reads through `step_ptr` see garbage.
            # Behind a pointer, the int lives outside any single call frame
            # and every access agrees on the same memory. Same issue as
            # `_ExecutorInner._q` in `gpu_async/executor.mojo`.
            var step = OwnedPointer(0)
            var step_ptr = step.unsafe_ptr[mut=True]().unsafe_origin_cast[
                MutUntrackedOrigin
            ]()

            var t1 = executor.add(_record_step(context, step_ptr))
            var t2 = executor.add(_record_step(context, step_ptr))
            var t3 = executor.add(_record_step(context, step_ptr))

            executor.wait()

            # Each task yields twice. If the executor truly round-robins —
            # rather than, say, draining one task to completion before
            # starting the next — every task gets its Nth segment before any
            # task gets its (N+1)th: t1, t2, t3 at steps 0-2, then t1, t2, t3
            # again at steps 3-5, then once more at steps 6-8, each in the
            # order the tasks were added.
            assert_equal(t1^.wait(), 36)  # segments at steps 0, 3, 6
            assert_equal(t2^.wait(), 147)  # segments at steps 1, 4, 7
            assert_equal(t3^.wait(), 258)  # segments at steps 2, 5, 8


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
