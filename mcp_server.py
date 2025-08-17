"""
MCP server exposing tools to control the remove-noise service programmatically.

Tools:
- denoise_path(path): submit a local file for denoising; returns task_id
- get_status(task_id): query current progress/status/details
- get_result(task_id): return local output file path and downloadable URL (if available)

This server reuses the functionality and in-memory task registry of api.py directly
(without starting the Flask HTTP server). It creates tasks and runs the same pipeline
in background threads, so status is consistent with the web server if both import the
same module in the same process. If the web server is running as a separate process,
this MCP server still works independently with its own in-memory registry.
"""
from __future__ import annotations

import os
import sys
import uuid
import threading
from pathlib import Path
from typing import Optional, Dict, Any

from fastmcp import FastMCP

# Ensure local imports resolve when launched from different working dirs
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the application logic from api.py
import api as app

mcp = FastMCP("remove-noise-mcp")


def _ensure_file_exists(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"File not found: {candidate}")
    return candidate


def _submit_task_for_path(file_path: Path) -> str:
    """Create a task in the app registry and submit background processing."""
    task_id = str(uuid.uuid4())
    app.update_task_progress(task_id, 0, "processing", "开始处理...")

    # Read bytes and re-use app.save_audio to place into TMPDIR and convert to 16k mono
    file_bytes = file_path.read_bytes()
    converted_path = app.save_audio(file_bytes, file_path.name, task_id)

    # Submit background processing using the app's executor
    def _worker():
        try:
            app.process_audio_async(converted_path, task_id, base_url=None)
        except Exception as exc:
            app.update_task_progress(task_id, 0, "failed", f"处理失败: {exc}")

    app.executor.submit(_worker)
    return task_id


@mcp.tool()
def denoise_path(path: str) -> Dict[str, Any]:
    """Submit a local audio/video file for denoising.

    Args:
        path: Absolute or relative file path to process.

    Returns: {"task_id": str, "message": str}
    """
    file_path = _ensure_file_exists(path)
    task_id = _submit_task_for_path(file_path)
    return {"task_id": task_id, "message": "任务已提交"}


@mcp.tool()
def get_status(task_id: str) -> Dict[str, Any]:
    """Query the current status/progress of a submitted task."""
    status = app.get_task_status(task_id)
    if status.get("status") == "not_found":
        return {"status": "not_found"}
    return status


def _guess_output_path_from_status(status: Dict[str, Any]) -> Optional[Path]:
    # Try to infer from result_url when present
    result_url = status.get("result_url")
    if result_url:
        try:
            name = Path(result_url).name
            candidate = Path(app.TMPDIR) / name
            if candidate.exists():
                return candidate
        except Exception:
            pass
    # Fallback: cannot infer path
    return None


@mcp.tool()
def get_result(task_id: str) -> Dict[str, Any]:
    """Return the output file path and download URL if the task completed."""
    status = app.get_task_status(task_id)
    if status.get("status") != "completed":
        return {"status": status.get("status", "unknown"), "message": status.get("message", "")}

    path = _guess_output_path_from_status(status)
    return {
        "status": "completed",
        "result_url": status.get("result_url"),
        "output_path": str(path) if path else None,
        "timestamp": status.get("timestamp"),
    }


if __name__ == "__main__":
    # Run MCP over stdio
    mcp.run()
