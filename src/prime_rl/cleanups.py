import logging
import shutil
import threading
import time
from pathlib import Path


CHECK_INTERVAL_SECONDS = 5
STEPS_TO_KEEP = 4


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_rollout_dirs() -> list[Path]:
    output_dir = get_repo_root() / "outputs"
    return [
        output_dir / "run_default" / "rollouts",
        output_dir / "rollouts",
    ]


def get_step_dirs(path: Path) -> list[tuple[int, Path]]:
    step_dirs: list[tuple[int, Path]] = []
    for child in path.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("step_"):
            continue
        try:
            step = int(child.name.split("_", maxsplit=1)[1])
        except ValueError:
            continue
        step_dirs.append((step, child))
    return sorted(step_dirs, key=lambda item: item[0], reverse=True)


def cleanup_rollout_dir(path: Path, steps_to_keep: int) -> None:
    if not path.exists():
        return

    step_dirs = get_step_dirs(path)
    if len(step_dirs) <= steps_to_keep:
        return

    for step, step_dir in step_dirs[steps_to_keep:]:
        shutil.rmtree(step_dir, ignore_errors=True)
        logging.info("Deleted old rollout step %s from %s", step, path)


class RolloutCleanupMonitor:
    def __init__(self, rollout_dirs: list[Path], steps_to_keep: int, check_interval_seconds: int) -> None:
        self.rollout_dirs = rollout_dirs
        self.steps_to_keep = steps_to_keep
        self.check_interval_seconds = check_interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logging.warning("Cleanup monitor is already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logging.info("Watching rollout directories: %s", [str(path) for path in self.rollout_dirs])

    def stop(self) -> None:
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self.check_interval_seconds + 1)
        logging.info("Cleanup monitor stopped")

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            for rollout_dir in self.rollout_dirs:
                try:
                    cleanup_rollout_dir(rollout_dir, self.steps_to_keep)
                except Exception:
                    logging.exception("Failed to clean %s", rollout_dir)
            self._stop_event.wait(self.check_interval_seconds)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    monitor = RolloutCleanupMonitor(
        rollout_dirs=get_rollout_dirs(),
        steps_to_keep=STEPS_TO_KEEP,
        check_interval_seconds=CHECK_INTERVAL_SECONDS,
    )
    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down cleanup monitor")
        monitor.stop()


if __name__ == "__main__":
    main()
