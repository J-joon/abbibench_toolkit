#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["tyro>=0.9"]
# ///
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Union

import tyro
from tyro import conf

CondaExe = Literal["conda", "mamba", "micromamba"]


def _parse_env_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"Invalid --env entry (expected KEY=VALUE): {s!r}")
        k, v = s.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --env entry (empty key): {s!r}")
        out[k] = v
    return out


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _conda_run_prefix(conda_exe: str, env: str, no_capture_output: bool) -> list[str]:
    base = [conda_exe, "run", "-n", env]
    if no_capture_output:
        base.append("--no-capture-output")
    return base


@dataclass(frozen=True)
class Run:
    """Run a command inside a conda environment (docker run-style)."""

    env: str
    cmd: Annotated[list[str], conf.arg(positional=True)]  # command after `--` or as positionals

    conda_exe: CondaExe = "conda"
    cwd: Path = Path(".")

    # Repeatable: --env KEY=VALUE
    env_vars: Annotated[list[str], conf.arg(name="env")] = ()
    inherit_host_env: bool = True

    no_capture_output: bool = True
    print_cmd: bool = True


@dataclass(frozen=True)
class Which:
    """Locate a program inside the environment."""

    env: str
    program: str
    conda_exe: CondaExe = "conda"
    no_capture_output: bool = True


Command = Union[Run, Which]


def main(c: Command) -> None:
    if isinstance(c, Run):
        if not c.cmd:
            raise SystemExit("No command provided. Example: ./envbox run myenv -- python -V")

        full_cmd = _conda_run_prefix(c.conda_exe, c.env, c.no_capture_output) + c.cmd

        child_env = os.environ.copy() if c.inherit_host_env else {}
        child_env.update(_parse_env_kv(list(c.env_vars)))

        if c.print_cmd:
            print(f"[envbox] env={c.env} cwd={c.cwd.resolve()}")
            print(f"[envbox] $ {_format_cmd(full_cmd)}")

        subprocess.run(full_cmd, cwd=str(c.cwd), env=child_env, check=True)
        return

    if isinstance(c, Which):
        full_cmd = _conda_run_prefix(c.conda_exe, c.env, c.no_capture_output) + [
            "python",
            "-c",
            "import shutil,sys; print(shutil.which(sys.argv[1]) or '')",
            c.program,
        ]
        subprocess.run(full_cmd, check=True)
        return

    raise AssertionError("Unhandled command type")


if __name__ == "__main__":
    main(tyro.cli(Command))
