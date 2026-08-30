#!/usr/bin/env python3
"""Symlinks this repo into a local `modular` checkout so its own Bazel rules
(mojo_library/mojo_test in src/BUILD.bazel and tests/BUILD.bazel) build and
test against a Mojo compiler, stdlib and `max` built from source there -- the
Mojo/Bazel equivalent of a Cargo `[patch]` override.

Run this once, and again if either checkout moves. $MODULAR_REPO must point
at your modular checkout -- there's no default, since where it's checked out
is entirely up to you:
    MODULAR_REPO=~/src/modular python3 scripts/setup_local_modular.py          # bash/zsh
    $env.MODULAR_REPO = "~/src/modular"; python3 scripts/setup_local_modular.py # nushell

Then, from inside the modular checkout (target pattern quoted so nushell
doesn't expand the trailing `...` into `../..`, its shorthand for "up two
directories" -- bash/zsh don't treat it specially, but quoting is harmless
there too):
    ./bazelw test '//max/_warp/tests/...'
    ./bazelw build //max/_warp/src:warp
"""

import os
import sys
from pathlib import Path

WARP_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    modular_repo_env = os.environ.get("MODULAR_REPO")
    if not modular_repo_env:
        sys.exit(
            "MODULAR_REPO is not set. Point it at your modular checkout, e.g.:\n"
            "  MODULAR_REPO=~/src/modular python3 scripts/setup_local_modular.py"
        )
    modular_repo = Path(modular_repo_env).expanduser().resolve()

    if not (modular_repo / "bazelw").exists():
        sys.exit(
            f"MODULAR_REPO={modular_repo} does not look like a modular checkout "
            "(no ./bazelw there)."
        )

    link = modular_repo / "max" / "_warp"
    link.unlink(missing_ok=True)
    link.symlink_to(WARP_ROOT, target_is_directory=True)

    # max/_warp is scratch (a symlink to this repo), not something to ever
    # commit into modular; excluded locally so it doesn't show up in its
    # `git status`.
    exclude_file = modular_repo / ".git" / "info" / "exclude"
    entry = "/max/_warp"
    if exclude_file.parent.is_dir():
        existing = exclude_file.read_text() if exclude_file.exists() else ""
        if entry not in existing.splitlines():
            with exclude_file.open("a") as f:
                f.write(f"\n{entry}\n")

    print(f"linked {link} -> {WARP_ROOT}")
    print(f"run: cd {modular_repo} && ./bazelw test '//max/_warp/tests/...'")


if __name__ == "__main__":
    main()
