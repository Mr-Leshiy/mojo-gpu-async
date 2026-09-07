#!/usr/bin/env python3
"""Runs the Mojo test suite: every tests/**/test_*.mojo file, in sorted order.

The hardware backends are compiled in only when their CPU features are enabled,
so the caller decides which ones to build via $MOJO_TEST_FEATURES: a
space-separated list of `mojo run` flags, e.g.
"--target-features=+neon,+aes,+sha2". CI sets it per-runner (see
.github/workflows/ci.yml). Nothing is defaulted here — when it is unset the
tests build for whatever `mojo run` autodetects on the host, which still covers
the naive backends.

Every test that needs a real accelerator -- not just CPU codegen -- lives in a
tests/test_with_gpu_*.mojo file, by convention. Set $MOJO_TEST_SKIP_GPU_TESTS
(to any non-empty value) to exclude those files on a host with no accelerator;
CI sets it on its free, GPU-less runners (see .github/workflows/ci.yml).

Every test file that's run happens even if an earlier one fails; the failures
are listed again at the end.
"""

import os
import subprocess
import sys
from itertools import filterfalse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _is_gpu_test(test: Path) -> bool:
    is_gpu = test.name.startswith("test_with_gpu_")
    if is_gpu:
        print(f"==> skipping {test.relative_to(ROOT)} (no accelerator)")
    return is_gpu


def main() -> int:
    # Split on whitespace so several flags can be passed at once; an unset or
    # empty value yields no arguments at all.
    features = os.environ.get("MOJO_TEST_FEATURES", "").split()
    tests = sorted(ROOT.glob("tests/**/test_*.mojo"))
    if not tests:
        print("no tests/**/test_*.mojo files found", file=sys.stderr)
        return 1

    if os.environ.get("MOJO_TEST_SKIP_GPU_TESTS"):
        tests = list(filterfalse(_is_gpu_test, tests))

    failed = []
    for test in tests:
        rel = test.relative_to(ROOT)
        cmd = ["mojo", "run", *features, "-I", ".", str(rel)]
        print(f"==> {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            failed.append(rel)

    if failed:
        print(f"\n{len(failed)} test file(s) failed:")
        for test in failed:
            print(f"  {test}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
