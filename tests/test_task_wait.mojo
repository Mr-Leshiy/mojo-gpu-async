"""`Task.wait()`: driving a single task's own coroutine to completion.

Unlike `Executor.wait()` (see `test_executor_wait.mojo`), `Task.wait()` only
guarantees progress on the task it's called on -- driving the shared
executor's queue is a side effect, not its contract. These tests exercise
that side effect and the values it returns, without depending on `Executor`
being drained through any other call first.
"""

from max.gpu.host import DeviceContext
from std.random import random_si64, seed
from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_false, assert_true

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

            # Neither task's `wait()` is preceded by `executor.wait()`: each
            # one has to drive the shared queue itself.
            assert_equal(first^.wait(), 7)
            assert_false(second.is_completed())
            assert_equal(second^.wait(), 12)


async def _yield_n_times(context: Context, n: Int) -> Int:
    var i = 0
    while i < n:
        await context.synchronize()
        i += 1
    return n


def test_task_yields_many_times_in_a_row_with_nothing_else_queued() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()

            # Only one task is ever queued, so every one of its several
            # yields resumes into an otherwise-empty queue -- there's
            # nothing else for the executor to interleave with in between.
            var task = executor.add(_yield_n_times(context, 8))

            assert_equal(task^.wait(), 8)


comptime _MAX_RANDOM_YIELDS = 4


def _spawn_and_wait_random_order[
    N: Int
](
    mut executor: Executor,
    context: Context,
    mut actual_total: Int,
    mut expected_total: Int,
) raises:
    """Queues `N` tasks, and for each one, randomly decides (at the point
    it's queued) whether to wait on it immediately or only after every task
    behind it has also been queued -- so the tasks are waited on in a
    genuinely randomized order relative to how they were created."""
    comptime if N == 0:
        pass
    else:
        var yields = Int(random_si64(0, _MAX_RANDOM_YIELDS))
        expected_total += yields
        var task = executor.add(_yield_n_times(context, yields))
        if random_si64(0, 1) == 0:
            actual_total += task^.wait()
            _spawn_and_wait_random_order[N - 1](
                executor, context, actual_total, expected_total
            )
        else:
            _spawn_and_wait_random_order[N - 1](
                executor, context, actual_total, expected_total
            )
            actual_total += task^.wait()


def test_task_wait_called_in_random_order() raises:
    comptime if has_accelerator():
        seed()
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()
            var actual_total = 0
            var expected_total = 0
            _spawn_and_wait_random_order[10](
                executor, context, actual_total, expected_total
            )
            assert_equal(actual_total, expected_total)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
