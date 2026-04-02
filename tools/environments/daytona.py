"""Daytona cloud execution environment.

Uses the Daytona Python SDK to run commands in cloud sandboxes.
Supports persistent sandboxes: when enabled, sandboxes are stopped on cleanup
and resumed on next creation, preserving the filesystem across sessions.
"""

import logging
import math
import os
import shlex
import threading
import uuid
import warnings
from pathlib import Path
from typing import Dict, Optional

from tools.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


class _DaytonaProcessHandle:
    """Adapter making Daytona's blocking SDK exec look like Popen."""

    def __init__(self, sandbox, cmd_string, cwd, timeout):
        self._done = threading.Event()
        self._returncode = None
        self._read_fd, self._write_fd = os.pipe()
        self.stdout = os.fdopen(self._read_fd, "r")
        self.stdin = None

        # Wrap with shell timeout (Daytona SDK timeout is unreliable)
        timed_cmd = f"timeout {timeout} bash -c {shlex.quote(cmd_string)}"

        def _run():
            try:
                response = sandbox.process.exec(timed_cmd, cwd=cwd)
                writer = os.fdopen(self._write_fd, "w")
                writer.write(response.result or "")
                writer.close()
                self._returncode = response.exit_code
            except Exception as e:
                try:
                    writer = os.fdopen(self._write_fd, "w")
                    writer.write(f"Daytona execution error: {e}")
                    writer.close()
                except Exception:
                    try:
                        os.close(self._write_fd)
                    except Exception:
                        pass
                self._returncode = 1
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll(self):
        return self._returncode if self._done.is_set() else None

    def kill(self):
        pass

    def wait(self, timeout=None):
        self._done.wait(timeout=timeout)
        return self._returncode

    @property
    def returncode(self):
        return self._returncode


class DaytonaEnvironment(BaseEnvironment):
    """Daytona cloud sandbox execution backend.

    Uses stopped/started sandbox lifecycle for filesystem persistence
    instead of snapshots, making it faster and stateless on the host.
    """

    def __init__(
        self,
        image: str,
        cwd: str = "/home/daytona",
        timeout: int = 60,
        cpu: int = 1,
        memory: int = 5120,       # MB (hermes convention)
        disk: int = 10240,        # MB (Daytona platform max is 10GB)
        persistent_filesystem: bool = True,
        task_id: str = "default",
    ):
        self._requested_cwd = cwd
        super().__init__(cwd=cwd, timeout=timeout)

        from daytona import (
            Daytona,
            CreateSandboxFromImageParams,
            DaytonaError,
            Resources,
            SandboxState,
        )

        self._persistent = persistent_filesystem
        self._task_id = task_id
        self._SandboxState = SandboxState
        self._daytona = Daytona()
        self._sandbox = None
        self._lock = threading.Lock()

        memory_gib = max(1, math.ceil(memory / 1024))
        disk_gib = max(1, math.ceil(disk / 1024))
        if disk_gib > 10:
            warnings.warn(
                f"Daytona: requested disk ({disk_gib}GB) exceeds platform limit (10GB). "
                f"Capping to 10GB. Set container_disk: 10240 in config to silence this.",
                stacklevel=2,
            )
            disk_gib = 10
        resources = Resources(cpu=cpu, memory=memory_gib, disk=disk_gib)

        labels = {"hermes_task_id": task_id}
        sandbox_name = f"hermes-{task_id}"

        # Try to resume an existing sandbox for this task
        if self._persistent:
            # 1. Try name-based lookup (new path)
            try:
                self._sandbox = self._daytona.get(sandbox_name)
                self._sandbox.start()
                logger.info("Daytona: resumed sandbox %s for task %s",
                            self._sandbox.id, task_id)
            except DaytonaError:
                self._sandbox = None
            except Exception as e:
                logger.warning("Daytona: failed to resume sandbox for task %s: %s",
                               task_id, e)
                self._sandbox = None

            # 2. Legacy fallback: find sandbox created before the naming migration
            if self._sandbox is None:
                try:
                    page = self._daytona.list(labels=labels, page=1, limit=1)
                    if page.items:
                        self._sandbox = page.items[0]
                        self._sandbox.start()
                        logger.info("Daytona: resumed legacy sandbox %s for task %s",
                                    self._sandbox.id, task_id)
                except Exception as e:
                    logger.debug("Daytona: no legacy sandbox found for task %s: %s",
                                 task_id, e)
                    self._sandbox = None

        # Create a fresh sandbox if we don't have one
        if self._sandbox is None:
            self._sandbox = self._daytona.create(
                CreateSandboxFromImageParams(
                    image=image,
                    name=sandbox_name,
                    labels=labels,
                    auto_stop_interval=0,
                    resources=resources,
                )
            )
            logger.info("Daytona: created sandbox %s for task %s",
                        self._sandbox.id, task_id)

        # Detect remote home dir first so mounts go to the right place.
        self._remote_home = "/root"
        try:
            home = self._sandbox.process.exec("echo $HOME").result.strip()
            if home:
                self._remote_home = home
                if self._requested_cwd in ("~", "/home/daytona"):
                    self.cwd = home
        except Exception:
            pass
        logger.info("Daytona: resolved home to %s, cwd to %s", self._remote_home, self.cwd)

        # Track synced files to avoid redundant uploads.
        # Key: remote_path, Value: (mtime, size)
        self._synced_files: Dict[str, tuple] = {}

        # Upload credential files and skills directory into the sandbox.
        self._sync_skills_and_credentials()

        # Capture login-shell environment into a snapshot for the unified model
        self.init_session()

    # ------------------------------------------------------------------
    # File sync
    # ------------------------------------------------------------------

    def _upload_if_changed(self, host_path: str, remote_path: str) -> bool:
        """Upload a file if its mtime/size changed since last sync."""
        hp = Path(host_path)
        try:
            stat = hp.stat()
            file_key = (stat.st_mtime, stat.st_size)
        except OSError:
            return False
        if self._synced_files.get(remote_path) == file_key:
            return False
        try:
            parent = str(Path(remote_path).parent)
            self._sandbox.process.exec(f"mkdir -p {parent}")
            self._sandbox.fs.upload_file(host_path, remote_path)
            self._synced_files[remote_path] = file_key
            return True
        except Exception as e:
            logger.debug("Daytona: upload failed %s: %s", host_path, e)
            return False

    def _sync_skills_and_credentials(self) -> None:
        """Upload changed credential files and skill files into the sandbox."""
        container_base = f"{self._remote_home}/.hermes"
        try:
            from tools.credential_files import get_credential_file_mounts, iter_skills_files

            for mount_entry in get_credential_file_mounts():
                remote_path = mount_entry["container_path"].replace("/root/.hermes", container_base, 1)
                if self._upload_if_changed(mount_entry["host_path"], remote_path):
                    logger.debug("Daytona: synced credential %s", remote_path)

            for entry in iter_skills_files(container_base=container_base):
                if self._upload_if_changed(entry["host_path"], entry["container_path"]):
                    logger.debug("Daytona: synced skill %s", entry["container_path"])
        except Exception as e:
            logger.debug("Daytona: could not sync skills/credentials: %s", e)

    def _ensure_sandbox_ready(self):
        """Restart sandbox if it was stopped (e.g., by a previous interrupt)."""
        self._sandbox.refresh_data()
        if self._sandbox.state in (self._SandboxState.STOPPED, self._SandboxState.ARCHIVED):
            self._sandbox.start()
            logger.info("Daytona: restarted sandbox %s", self._sandbox.id)

    # ------------------------------------------------------------------
    # Unified execution hooks
    # ------------------------------------------------------------------

    def _before_execute(self) -> None:
        """Ensure sandbox is ready and sync credentials before each command."""
        with self._lock:
            self._ensure_sandbox_ready()
        self._sync_skills_and_credentials()

    def _run_bash(self, cmd_string: str, *, stdin_data: str | None = None):
        """Spawn ``bash -c <cmd_string>`` inside the Daytona sandbox.

        Returns a _DaytonaProcessHandle (satisfies the ProcessHandle protocol).
        stdin_data is embedded as a heredoc since Daytona cannot pipe stdin.
        """
        if stdin_data is not None:
            marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            while marker in stdin_data:
                marker = f"HERMES_EOF_{uuid.uuid4().hex[:8]}"
            cmd_string = f"{cmd_string} << '{marker}'\n{stdin_data}\n{marker}"
        effective_cwd = self.cwd or None
        return _DaytonaProcessHandle(
            self._sandbox, cmd_string, effective_cwd, self.timeout,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self):
        with self._lock:
            if self._sandbox is None:
                return
            try:
                if self._persistent:
                    self._sandbox.stop()
                    logger.info("Daytona: stopped sandbox %s (filesystem preserved)",
                                self._sandbox.id)
                else:
                    self._daytona.delete(self._sandbox)
                    logger.info("Daytona: deleted sandbox %s", self._sandbox.id)
            except Exception as e:
                logger.warning("Daytona: cleanup failed: %s", e)
            self._sandbox = None
