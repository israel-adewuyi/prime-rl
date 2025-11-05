import shutil
import threading
import time
from pathlib import Path
from typing import List, Tuple

from loguru import logger


class FileMonitor:
    """Monitors directories and keeps only the most recent N files."""

    def __init__(self, directories: List[Tuple[str, int]], check_interval: int = 5):
        """
        Initialize the file monitor.

        Args:
            directories: List of tuples (directory_path, max_files_to_keep)
            check_interval: Seconds between checks
        """
        self.directories = directories
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start monitoring in a separate thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("File monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"File monitor started with {len(self.directories)} directories")

    def stop(self):
        """Stop the monitoring thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=self.check_interval + 1)
        logger.info("File monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop that runs in the thread."""
        while not self._stop_event.is_set():
            for directory, max_files in self.directories:
                try:
                    self._cleanup_directory(directory, max_files)
                except Exception as e:
                    logger.error(f"Error cleaning directory {directory}: {e}")

            self._stop_event.wait(self.check_interval)

    def _cleanup_directory(self, directory: str, max_files: int):
        """Clean up old files in a directory, keeping only the most recent ones."""
        dir_path = Path(directory)

        if not dir_path.exists():
            return

        # Get all files in the directory (not subdirectories)
        files = [f for f in dir_path.iterdir()]

        if len(files) <= max_files:
            logger.info(files)
            return

        # Sort by modification time (most recent first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Files to delete are those beyond the max_files limit
        files_to_delete = files[max_files:]

        for item_path in files_to_delete:
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                    logger.info(f"Deleted old directory: {item_path}")
                else:
                    item_path.unlink()
                    logger.info(f"Deleted old file: {item_path}")
            except Exception as e:
                logger.error(f"Failed to delete {item_path}: {e}")


# Example usage
if __name__ == "__main__":
    # Configure directories to monitor with their retention policies
    directories_to_monitor = [
        ("./outputs/weights", 3),  # Keep last 3 weight files
        ("./outputs/rollouts", 3),  # Keep last 10 rollout files
    ]

    # Create and start monitor
    monitor = FileMonitor(
        directories=directories_to_monitor,
        check_interval=5,  # Check every 5 seconds
    )

    monitor.start()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        monitor.stop()
