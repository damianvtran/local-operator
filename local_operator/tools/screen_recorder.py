import asyncio
import errno
import json
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import uuid
from asyncio.subprocess import DEVNULL, PIPE, Process
from pathlib import Path
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)

# Directory for temporary files and state
TEMP_DIR_BASE = Path.home() / ".local-operator" / "tmp" / "recordings"
STATE_FILE_PATH = TEMP_DIR_BASE / "active_recordings.json"

# Lock for serializing access to the state file from concurrent asyncio tasks
_state_accessor_lock = asyncio.Lock()


class ToolError(Exception):
    """Custom exception for recording tool errors."""


# Information stored in the JSON state file
class StoredRecordingInfo(TypedDict):
    pid: int
    output_path: str
    temp_output_path: str
    stderr_log_path: str
    record_video: bool
    record_audio: bool


# --- State File Management ---
async def _read_state_file() -> Dict[str, StoredRecordingInfo]:
    async with _state_accessor_lock:
        if not STATE_FILE_PATH.exists():
            return {}
        try:
            with open(STATE_FILE_PATH, "r") as f:
                content = f.read()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not read or decode state file, treating as empty.", exc_info=True)
            return {}


async def _write_state_file(data: Dict[str, StoredRecordingInfo]) -> None:
    async with _state_accessor_lock:
        STATE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE_PATH, "w") as f:
            json.dump(data, f, indent=4)


# --- FFmpeg Process Helpers ---
async def _run_ffmpeg_command(command: List[str], process_name: str, stderr_handle) -> Process:
    """
    Start an FFmpeg subprocess asynchronously.

    Args:
        command: FFmpeg command and arguments.
        process_name: Identifier for logging.
        stderr_handle: File handle for FFmpeg's stderr.

    Returns:
        The FFmpeg subprocess.

    Raises:
        ToolError: If the subprocess cannot be started.
    """
    logger.info("Launching %s: %s", process_name, " ".join(command))
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=PIPE,
            stdout=DEVNULL,
            stderr=stderr_handle,
        )
    except Exception as exc:
        logger.exception("Failed to start FFmpeg process %s", process_name)
        if stderr_handle:
            try:
                stderr_handle.close()
            except Exception:
                pass
        raise ToolError(f"Could not start FFmpeg process '{process_name}': {exc}") from exc
    return process


async def _wait_for_pid(pid: int, timeout_seconds: float = 5.0) -> bool:
    """
    Waits for a process with the given PID to terminate.
    Returns True if terminated, False if timeout.
    """
    elapsed = 0.0
    interval = 0.1
    while elapsed < timeout_seconds:
        try:
            os.kill(pid, 0)
            await asyncio.sleep(interval)
            elapsed += interval
        except OSError as e:
            if e.errno == errno.ESRCH:
                logger.debug(f"Process {pid} confirmed terminated (ESRCH).")
                return True
            if e.errno == errno.EPERM:
                logger.warning(f"Permission error checking PID {pid}; assuming it's running.")
                await asyncio.sleep(interval)
                elapsed += interval
                continue
            logger.error(f"Unexpected OSError checking PID {pid}: {e}")
            raise
    logger.warning(f"Process {pid} did not terminate within {timeout_seconds}s.")
    return False


# --- Device Detection for macOS avfoundation ---
async def _find_avfoundation_device_index(device_type: str, hint: str) -> int:
    """
    List avfoundation devices and return index of the first device whose name contains the hint.
    device_type: 'video' or 'audio'.
    """
    cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=DEVNULL, stdout=DEVNULL, stderr=PIPE
        )
        _, stderr = await proc.communicate()
        output = stderr.decode(errors="ignore")
    except Exception as e:
        logger.warning("Could not list avfoundation devices: %s", e)
        return 1 if device_type == "video" else 0

    found_index = None
    lines = output.splitlines()
    scanning = False
    for line in lines:
        if device_type == "video" and "AVFoundation video devices" in line:
            scanning = True
            continue
        if device_type == "audio" and "AVFoundation audio devices" in line:
            scanning = True
            continue
        if scanning:
            m = re.search(r"\[([0-9]+)\]\s*(.+)", line)
            if m:
                idx = int(m.group(1))
                name = m.group(2).strip()
                if hint.lower() in name.lower():
                    found_index = idx
                    break
    return found_index if found_index is not None else (1 if device_type == "video" else 0)


# --- OS Specific FFmpeg Arguments ---
async def _get_os_specific_video_input(video_source: str) -> List[str]:
    system = platform.system()
    if video_source == "screen":
        if system == "Darwin":
            idx = await _find_avfoundation_device_index("video", "capture screen")
            return [
                "-f",
                "avfoundation",
                "-thread_queue_size",
                "512",
                "-probesize",
                "10M",
                "-framerate",
                "30",
                "-i",
                str(idx),
            ]
        if system == "Windows":
            return ["-f", "gdigrab", "-framerate", "30", "-i", "desktop"]
        if system == "Linux":
            return [
                "-f",
                "x11grab",
                "-thread_queue_size",
                "512",
                "-probesize",
                "10M",
                "-draw_mouse",
                "1",
                "-s",
                "1920x1080",
                "-i",
                ":0.0",
            ]
        raise NotImplementedError(f"Screen capture not implemented for {system}")
    if video_source == "window":
        raise NotImplementedError("Window-specific recording is not supported yet.")
    raise ValueError(f"Unsupported video_source: {video_source}")


async def _get_os_specific_audio_input(audio_source: str) -> List[str]:
    system = platform.system()
    if audio_source == "system" and system == "Darwin":
        logger.warning(
            "System audio capture on macOS may require a virtual driver (e.g., BlackHole). "
            "Falling back to default microphone input."
        )
        audio_source = "microphone"
    if audio_source == "system":
        if system == "Windows":
            return ["-f", "dshow", "-i", "audio=virtual-audio-capturer"]
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        if system == "Darwin":
            idx = await _find_avfoundation_device_index("audio", "microphone")
            return ["-f", "avfoundation", "-thread_queue_size", "512", "-i", f":{idx}"]
    if audio_source == "microphone":
        if system == "Windows":
            return ["-f", "dshow", "-i", "audio=Microphone"]
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        if system == "Darwin":
            idx = await _find_avfoundation_device_index("audio", "microphone")
            return ["-f", "avfoundation", "-thread_queue_size", "512", "-i", f":{idx}"]
    raise ValueError(f"Unsupported audio_source: {audio_source}")


# --- Public Tool Functions ---
async def start_recording_tool(
    output_path: str,
    record_audio: bool = True,
    record_video: bool = True,
    audio_source: str = "system",
    video_source: str = "screen",
) -> str:
    if shutil.which("ffmpeg") is None:
        raise ToolError("FFmpeg executable not found. Please install FFmpeg.")
    if not (record_audio or record_video):
        raise ValueError("At least one of record_audio or record_video must be True.")

    rec_id = str(uuid.uuid4())
    TEMP_DIR_BASE.mkdir(parents=True, exist_ok=True)

    temp_output = TEMP_DIR_BASE / f"{rec_id}_temp.mp4"
    stderr_log_path = TEMP_DIR_BASE / f"{rec_id}_stderr.log"
    final_output = Path(output_path).expanduser().resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)

    # Determine video codec availability
    video_codec = "libx264"
    try:
        encoders = subprocess.check_output(["ffmpeg", "-encoders"], stderr=DEVNULL).decode(
            errors="ignore"
        )
        if " libx264 " not in encoders:
            video_codec = "h264"
            logger.warning("libx264 encoder not found; falling back to h264")
    except Exception:
        logger.warning("Could not check ffmpeg encoders; using libx264 by default")

    # Build ffmpeg command: inputs first, then codec/output options
    cmd: List[str] = ["ffmpeg", "-y"]

    # On macOS, combine audio and video in a single avfoundation input
    if platform.system() == "Darwin" and record_video and record_audio:
        v_idx = await _find_avfoundation_device_index("video", "capture screen")
        a_idx = await _find_avfoundation_device_index(
            "audio", "microphone" if audio_source == "microphone" else ""
        )
        cmd += [
            "-f",
            "avfoundation",
            "-thread_queue_size",
            "512",
            "-probesize",
            "10M",
            "-framerate",
            "30",
            "-i",
            f"{v_idx}:{a_idx}",
        ]
    else:
        if record_video:
            cmd += await _get_os_specific_video_input(video_source)
        if record_audio:
            cmd += await _get_os_specific_audio_input(audio_source)

    # Codec and output options
    if record_video:
        cmd += ["-c:v", video_codec, "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"]
    if record_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]

    # Output file
    cmd.append(str(temp_output))

    proc_name = (
        "CombinedRecorder"
        if record_audio and record_video
        else "VideoRecorder" if record_video else "AudioRecorder"
    )

    stderr_file_handle = None
    try:
        stderr_file_handle = open(stderr_log_path, "wb")
        process = await _run_ffmpeg_command(cmd, proc_name, stderr_handle=stderr_file_handle)
    except Exception:
        if stderr_file_handle:
            try:
                stderr_file_handle.close()
            except Exception:
                pass
            if stderr_log_path.exists():
                try:
                    stderr_log_path.unlink()
                except Exception:
                    logger.warning(
                        f"Could not delete stderr log {stderr_log_path} after ffmpeg start failure."
                    )
        raise

    stored_info: StoredRecordingInfo = {
        "pid": process.pid,
        "output_path": str(final_output),
        "temp_output_path": str(temp_output),
        "stderr_log_path": str(stderr_log_path),
        "record_video": record_video,
        "record_audio": record_audio,
    }

    all_states = await _read_state_file()
    all_states[rec_id] = stored_info
    await _write_state_file(all_states)

    logger.info(
        f"Recording started with ID {rec_id}, PID {process.pid}. "
        f"Temp: {temp_output}, Output: {final_output}"
    )
    return rec_id


async def stop_recording_tool(recording_id: str) -> str:
    all_states = await _read_state_file()
    info = all_states.pop(recording_id, None)

    if info is None:
        raise ToolError(f"No active recording with ID '{recording_id}' found in state file.")

    await _write_state_file(all_states)

    pid = info["pid"]
    temp_output_path = Path(info["temp_output_path"])
    final_output_path = Path(info["output_path"])
    stderr_log_path = Path(info["stderr_log_path"])

    logger.info(f"Attempting to stop recording ID {recording_id} (PID: {pid})")

    terminated_gracefully = False
    try:
        if not await _wait_for_pid(pid, timeout_seconds=0.1):
            os.kill(pid, signal.SIGINT)
            terminated_gracefully = await _wait_for_pid(pid, timeout_seconds=7.0)
            if not terminated_gracefully:
                os.kill(pid, signal.SIGTERM)
                terminated_gracefully = await _wait_for_pid(pid, timeout_seconds=3.0)
                if not terminated_gracefully:
                    os.kill(pid, signal.SIGKILL)
                    terminated_gracefully = await _wait_for_pid(pid, timeout_seconds=1.0)
    except ProcessLookupError:
        terminated_gracefully = True
    except Exception as e:
        logger.exception(f"Error while stopping FFmpeg process {pid}: {e}")

    logger.info(
        f"FFmpeg process PID {pid} termination status: "
        f"{'Graceful/Confirmed' if terminated_gracefully else 'Forced/Uncertain'}"
    )

    # Log stderr
    if stderr_log_path.exists():
        try:
            stderr_content = stderr_log_path.read_text(errors="ignore")
            if stderr_content:
                logger.info(f"FFmpeg stderr for {recording_id}:\n{stderr_content}")
        except Exception as e:
            logger.warning(f"Could not read stderr log {stderr_log_path}: {e}")

    # Wait for temp file flush
    temp_file_found = False
    for _ in range(10):
        if temp_output_path.exists() and temp_output_path.stat().st_size > 0:
            temp_file_found = True
            break
        await asyncio.sleep(0.2)

    if not temp_file_found:
        error_msg = f"Temporary file missing or empty: {temp_output_path}"
        if stderr_log_path.exists():
            stderr_excerpt = stderr_log_path.read_text(errors="ignore")[:500]
            error_msg += f"\nFFmpeg stderr excerpt:\n{stderr_excerpt}"
        logger.error(error_msg)
        if stderr_log_path.exists():
            try:
                stderr_log_path.unlink()
            except Exception:
                logger.warning(f"Could not delete stderr log {stderr_log_path}")
        raise ToolError(error_msg)

    # Move to final output
    try:
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(temp_output_path), str(final_output_path))
        except OSError as exc:
            if exc.errno == errno.EXDEV:
                shutil.move(str(temp_output_path), str(final_output_path))
            else:
                raise
        logger.info(f"Recording {recording_id} saved to {final_output_path}")
    except Exception as exc:
        logger.exception(f"Failed to move recording {recording_id} to final destination: {exc}")
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                logger.warning(f"Could not delete temp file {temp_output_path}")
        if stderr_log_path.exists():
            try:
                stderr_log_path.unlink()
            except Exception:
                logger.warning(f"Could not delete stderr log {stderr_log_path}")
        raise ToolError(f"Failed to finalize recording {recording_id}: {exc}")

    # Cleanup stderr log
    if stderr_log_path.exists():
        stderr_log_path.unlink()

    return f"Recording saved to: {final_output_path}"
