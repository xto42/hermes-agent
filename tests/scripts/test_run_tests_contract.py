"""Contract tests for the canonical test runner.

The runner is itself part of the runtime quality gate. It may verify that the
selected environment has the required test dependencies, but it must not mutate
that environment while trying to prove the checkout is healthy.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "run_tests.sh"


def _copy_runner_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    runner = scripts / "run_tests.sh"
    shutil.copy2(RUNNER, runner)
    runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
    return repo


def test_runner_fails_fast_when_pytest_split_missing_without_pip_install(tmp_path: Path) -> None:
    repo = _copy_runner_fixture(tmp_path)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    calls_file = tmp_path / "python_calls.log"

    (venv_bin / "activate").write_text("# fake activate for runner contract test\n", encoding="utf-8")

    fake_python = venv_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$PYTHON_CALLS_FILE\"\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "  # Simulate missing pytest_split.\n"
        "  exit 1\n"
        "fi\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pip\" ]; then\n"
        "  echo 'pip install must not be called by scripts/run_tests.sh' >&2\n"
        "  exit 42\n"
        "fi\n"
        "exit 99\n",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PYTHON_CALLS_FILE": str(calls_file),
    }
    result = subprocess.run(
        [str(repo / "scripts" / "run_tests.sh"), "tests/example_test.py"],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "pytest-split is not installed" in result.stderr
    assert "pip install" not in calls_file.read_text(encoding="utf-8")
    assert "python -m pip install" not in result.stdout
