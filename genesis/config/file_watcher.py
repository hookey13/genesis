"""File Watcher Module for Configuration Hot-Reload.

This module provides file system monitoring capabilities to detect changes
in configuration files and trigger automatic reloads without service interruption.
"""

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class FileWatcher:
    """Watches configuration files for changes and triggers reload callbacks."""

    def __init__(
        self,
        watch_paths: list[Path],
        poll_interval: float = 1.0,
        debounce_seconds: float = 0.5,
    ):
        """Initialize the file watcher.

        Args:
            watch_paths: List of paths to watch (files or directories)
            poll_interval: Seconds between file system polls
            debounce_seconds: Seconds to wait before processing changes
        """
        self.watch_paths = [Path(p) for p in watch_paths]
        self.poll_interval = poll_interval
        self.debounce_seconds = debounce_seconds

        # File tracking
        self.file_states: dict[str, dict[str, Any]] = {}
        self.pending_changes: set[str] = set()

        # Callbacks
        self.change_callbacks: list[Callable] = []
        self.error_callbacks: list[Callable] = []

        # Control flags
        self._running = False
        self._watch_task: asyncio.Task | None = None
        self._debounce_task: asyncio.Task | None = None

        logger.info(
            "FileWatcher initialized",
            watch_paths=[str(p) for p in watch_paths],
            poll_interval=poll_interval,
            debounce_seconds=debounce_seconds,
        )

    async def start(self) -> None:
        """Start watching for file changes."""
        if self._running:
            logger.warning("FileWatcher already running")
            return

        self._running = True

        # Initialize file states
        await self._scan_files()

        # Start watch loop
        self._watch_task = asyncio.create_task(self._watch_loop())

        logger.info("FileWatcher started")

    async def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._running:
            logger.warning("FileWatcher not running")
            return

        self._running = False

        # Cancel tasks
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        if self._debounce_task:
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass
            self._debounce_task = None

        logger.info("FileWatcher stopped")

    async def _scan_files(self) -> None:
        """Scan all watched paths and initialize file states."""
        for watch_path in self.watch_paths:
            if watch_path.is_file():
                await self._update_file_state(watch_path)
            elif watch_path.is_dir():
                # Scan directory for relevant files
                for pattern in ["*.yaml", "*.yml", "*.json"]:
                    for file_path in watch_path.glob(pattern):
                        await self._update_file_state(file_path)

    async def _update_file_state(self, file_path: Path) -> bool:
        """Update the state of a single file.

        Args:
            file_path: Path to the file

        Returns:
            True if file changed, False otherwise
        """
        str_path = str(file_path)

        try:
            if not file_path.exists():
                # File deleted
                if str_path in self.file_states:
                    del self.file_states[str_path]
                    logger.info(f"File deleted: {str_path}")
                    return True
                return False

            # Get file stats
            stat = file_path.stat()
            mtime = stat.st_mtime
            size = stat.st_size

            # Calculate file hash for content comparison
            file_hash = await self._calculate_file_hash(file_path)

            # Check if file is new or changed
            if str_path not in self.file_states:
                # New file
                self.file_states[str_path] = {
                    "mtime": mtime,
                    "size": size,
                    "hash": file_hash,
                    "last_checked": datetime.now(UTC),
                }
                logger.info(f"New file detected: {str_path}")
                return True

            # Check for changes
            old_state = self.file_states[str_path]
            changed = False

            if old_state["hash"] != file_hash:
                changed = True
                logger.info(
                    "File content changed",
                    file=str_path,
                    old_hash=old_state["hash"][:8],
                    new_hash=file_hash[:8],
                )
            elif old_state["mtime"] != mtime or old_state["size"] != size:
                # Metadata changed but content same (maybe touch command)
                logger.debug(
                    "File metadata changed but content unchanged", file=str_path
                )

            # Update state
            self.file_states[str_path] = {
                "mtime": mtime,
                "size": size,
                "hash": file_hash,
                "last_checked": datetime.now(UTC),
            }

            return changed

        except Exception as e:
            logger.error("Error checking file state", file=str_path, error=str(e))
            return False

    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.sha256()

        try:
            # Read file in chunks to handle large files
            with open(file_path, "rb") as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)

            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(
                "Error calculating file hash", file=str(file_path), error=str(e)
            )
            return ""

    async def _watch_loop(self) -> None:
        """Main watch loop that polls for file changes."""
        logger.info("Starting file watch loop")

        while self._running:
            try:
                # Check all watched paths
                changes_detected = False

                for watch_path in self.watch_paths:
                    if watch_path.is_file():
                        if await self._update_file_state(watch_path):
                            self.pending_changes.add(str(watch_path))
                            changes_detected = True
                    elif watch_path.is_dir():
                        # Check all relevant files in directory
                        for pattern in ["*.yaml", "*.yml", "*.json"]:
                            for file_path in watch_path.glob(pattern):
                                if await self._update_file_state(file_path):
                                    self.pending_changes.add(str(file_path))
                                    changes_detected = True

                # Check for deleted files
                current_files = set()
                for watch_path in self.watch_paths:
                    if watch_path.is_file():
                        current_files.add(str(watch_path))
                    elif watch_path.is_dir():
                        for pattern in ["*.yaml", "*.yml", "*.json"]:
                            for file_path in watch_path.glob(pattern):
                                current_files.add(str(file_path))

                deleted_files = set(self.file_states.keys()) - current_files
                for deleted_file in deleted_files:
                    logger.info(f"File deleted: {deleted_file}")
                    del self.file_states[deleted_file]
                    self.pending_changes.add(deleted_file)
                    changes_detected = True

                # Trigger debounced processing if changes detected
                if changes_detected and self.pending_changes:
                    await self._trigger_debounced_processing()

                # Sleep before next poll
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error("Error in watch loop", error=str(e))
                await self._notify_error_callbacks(e)
                await asyncio.sleep(self.poll_interval)

    async def _trigger_debounced_processing(self) -> None:
        """Trigger debounced processing of pending changes."""
        # Cancel existing debounce task if any
        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()
            try:
                await self._debounce_task
            except asyncio.CancelledError:
                pass

        # Create new debounce task
        self._debounce_task = asyncio.create_task(self._process_pending_changes())

    async def _process_pending_changes(self) -> None:
        """Process pending file changes after debounce period."""
        # Wait for debounce period
        await asyncio.sleep(self.debounce_seconds)

        if not self.pending_changes:
            return

        # Process all pending changes
        changed_files = list(self.pending_changes)
        self.pending_changes.clear()

        logger.info(
            "Processing file changes",
            files_count=len(changed_files),
            files=changed_files,
        )

        # Notify callbacks
        await self._notify_change_callbacks(changed_files)

    async def _notify_change_callbacks(self, changed_files: list[str]) -> None:
        """Notify registered callbacks of file changes.

        Args:
            changed_files: List of changed file paths
        """
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(changed_files)
                else:
                    callback(changed_files)
            except Exception as e:
                logger.error(
                    "Error in change callback", callback=callback.__name__, error=str(e)
                )
                await self._notify_error_callbacks(e)

    async def _notify_error_callbacks(self, error: Exception) -> None:
        """Notify registered error callbacks.

        Args:
            error: Exception that occurred
        """
        for callback in self.error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error)
                else:
                    callback(error)
            except Exception as e:
                logger.error(
                    "Error in error callback", callback=callback.__name__, error=str(e)
                )

    def register_change_callback(self, callback: Callable) -> None:
        """Register a callback for file changes.

        Args:
            callback: Function to call when files change
        """
        self.change_callbacks.append(callback)
        logger.info(f"Registered change callback: {callback.__name__}")

    def unregister_change_callback(self, callback: Callable) -> None:
        """Unregister a change callback.

        Args:
            callback: Callback to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            logger.info(f"Unregistered change callback: {callback.__name__}")

    def register_error_callback(self, callback: Callable) -> None:
        """Register a callback for errors.

        Args:
            callback: Function to call on errors
        """
        self.error_callbacks.append(callback)
        logger.info(f"Registered error callback: {callback.__name__}")

    def unregister_error_callback(self, callback: Callable) -> None:
        """Unregister an error callback.

        Args:
            callback: Callback to remove
        """
        if callback in self.error_callbacks:
            self.error_callbacks.remove(callback)
            logger.info(f"Unregistered error callback: {callback.__name__}")

    def get_watched_files(self) -> list[str]:
        """Get list of currently watched files.

        Returns:
            List of file paths being watched
        """
        return list(self.file_states.keys())

    def get_file_state(self, file_path: str) -> dict[str, Any] | None:
        """Get state information for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            File state dictionary or None if not watched
        """
        return self.file_states.get(file_path)

    def is_running(self) -> bool:
        """Check if file watcher is running.

        Returns:
            True if running, False otherwise
        """
        return self._running

    async def force_check(self) -> list[str]:
        """Force an immediate check for file changes.

        Returns:
            List of changed files detected
        """
        changed_files = []

        for watch_path in self.watch_paths:
            if watch_path.is_file():
                if await self._update_file_state(watch_path):
                    changed_files.append(str(watch_path))
            elif watch_path.is_dir():
                for pattern in ["*.yaml", "*.yml", "*.json"]:
                    for file_path in watch_path.glob(pattern):
                        if await self._update_file_state(file_path):
                            changed_files.append(str(file_path))

        if changed_files:
            logger.info(
                "Force check detected changes",
                files_count=len(changed_files),
                files=changed_files,
            )
            await self._notify_change_callbacks(changed_files)

        return changed_files

    def get_stats(self) -> dict[str, Any]:
        """Get file watcher statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "is_running": self._running,
            "watched_paths": [str(p) for p in self.watch_paths],
            "watched_files_count": len(self.file_states),
            "pending_changes_count": len(self.pending_changes),
            "poll_interval": self.poll_interval,
            "debounce_seconds": self.debounce_seconds,
            "registered_callbacks": len(self.change_callbacks),
            "registered_error_callbacks": len(self.error_callbacks),
        }


class ConfigFileWatcher(FileWatcher):
    """Specialized file watcher for configuration files with reload support."""

    def __init__(
        self,
        config_manager,
        config_path: Path,
        poll_interval: float = 1.0,
        debounce_seconds: float = 0.5,
    ):
        """Initialize configuration file watcher.

        Args:
            config_manager: StrategyConfigManager instance
            config_path: Path to configuration directory
            poll_interval: Seconds between file system polls
            debounce_seconds: Seconds to wait before processing changes
        """
        super().__init__(
            watch_paths=[config_path],
            poll_interval=poll_interval,
            debounce_seconds=debounce_seconds,
        )

        self.config_manager = config_manager

        # Register auto-reload callback
        self.register_change_callback(self._handle_config_changes)

    async def _handle_config_changes(self, changed_files: list[str]) -> None:
        """Handle configuration file changes.

        Args:
            changed_files: List of changed file paths
        """
        for file_path in changed_files:
            path = Path(file_path)

            # Skip non-config files
            if path.suffix not in [".yaml", ".yml", ".json"]:
                continue

            # Extract strategy name from filename
            # Expected format: {strategy_name}.{yaml|yml|json}
            strategy_name = path.stem

            logger.info(
                "Reloading configuration", file=file_path, strategy=strategy_name
            )

            try:
                # Reload configuration
                if path.exists():
                    await self.config_manager.load_config_file(path)
                else:
                    # File was deleted, remove config
                    if strategy_name in self.config_manager.configs:
                        del self.config_manager.configs[strategy_name]
                        logger.info("Configuration removed", strategy=strategy_name)

            except Exception as e:
                logger.error(
                    "Failed to reload configuration",
                    file=file_path,
                    strategy=strategy_name,
                    error=str(e),
                )
