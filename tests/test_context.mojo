from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from warp.context import Context
from warp.executor import Executor


async def _yield_once(context: Context) -> Int:
    await context.synchronize()
    return 1


def test_awaiting_context_from_a_different_executor_does_not_complete_the_task() raises:
    with DeviceContext() as ctx:
        var executor1 = Executor(ctx)
        var executor2 = Executor(ctx)
        var context2 = executor2.context()

        # Queued on executor1, but the coroutine awaits a `Context` that
        # belongs to executor2 -- its yield re-queues it onto executor2,
        # not executor1 (see `Context.synchronize`'s docstring).
        var task = executor1.add(_yield_once(context2))

        executor1.wait()
        # Migrated to the other executor's queue: executor1 draining its
        # own (now-empty) queue does not complete it.
        assert_false(task.is_completed())

        executor2.wait()
        assert_true(task.is_completed())
        assert_equal(task^.wait(), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
