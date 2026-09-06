"""`RaisingTask`: error propagation from a coroutine that raises
immediately, after suspending, or from a nested coroutine.
"""

from max.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.testing import TestSuite, assert_raises

from warp.context import Context
from warp.executor import Executor


async def _raises_immediately() raises -> Int:
    raise Error("immediate failure")


def test_raising_task_raises_immediately() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var task = executor.add(_raises_immediately())
            with assert_raises(contains="immediate failure"):
                _ = task^.wait()


async def _raises_after_yields(context: Context) raises -> Int:
    await context.synchronize()
    await context.synchronize()
    raise Error("failure after yields")


def test_raising_task_raises_after_yields() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()
            var task = executor.add(_raises_after_yields(context))
            with assert_raises(contains="failure after yields"):
                _ = task^.wait()


async def _raises_from_nested_coroutine(context: Context) raises -> Int:
    async def _raises_inner(context: Context) raises -> Int:
        await context.synchronize()
        raise Error("nested failure")

    return await _raises_inner(context)


def test_raising_task_raises_from_nested_coroutine() raises:
    comptime if has_accelerator():
        with DeviceContext() as ctx:
            var executor = Executor(ctx)
            var context = executor.context()
            var task = executor.add(_raises_from_nested_coroutine(context))
            with assert_raises(contains="nested failure"):
                _ = task^.wait()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
