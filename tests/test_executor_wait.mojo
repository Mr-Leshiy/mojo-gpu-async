from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_false

from warp.executor import Executor


# Neither test here drives a task to completion via a real yield -- what's
# under test is `Executor.wait()`'s own contract (empty queue, called more
# than once, an executor torn down before draining), not any particular
# task shape. A non-yielding task keeps that decoupled from `Context` and
# its one suspend point, `Context.synchronize()`.
async def _no_yield(value: Int) -> Int:
    return value


def test_executor_wait_on_empty_queue_and_called_twice() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)

        executor.wait()
        executor.wait()

        var task = executor.add(_no_yield(1))
        executor.wait()
        # Nothing left queued -- must not hang or error.
        executor.wait()

        assert_equal(task^.wait(), 1)


def test_executor_dropped_with_task_still_queued() raises:
    with DeviceContext() as ctx:
        var executor = Executor(ctx)

        # Queued but never driven: neither `task.wait()` nor
        # `executor.wait()` is called before both go out of scope.
        var task = executor.add(_no_yield(1))
        assert_false(task.is_completed())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
