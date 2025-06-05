import asyncio
import errno
import json
import logging
import os
import platform
import shutil
import signal
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
                if not content:  # Handle empty file
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
            stdin=PIPE,  # For sending 'q' if possible, though PID stop won't use it
            stdout=DEVNULL,
            stderr=stderr_handle,
        )
    except Exception as exc:
        logger.exception("Failed to start FFmpeg process %s", process_name)
        if stderr_handle:
            try:
                stderr_handle.close()
            except Exception:
                pass  # Ignore errors during cleanup of stderr handle
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
            os.kill(pid, 0)  # Check if process exists; raises OSError if not (or permission denied)
            await asyncio.sleep(interval)
            elapsed += interval
        except OSError as e:
            if e.errno == errno.ESRCH:  # ESRCH: No such process
                logger.debug(f"Process {pid} confirmed terminated (ESRCH).")
                return True
            # If permission denied (EPERM), process might exist but we can't signal it.
            # Treat as still running for polling, but log it.
            if e.errno == errno.EPERM:
                logger.warning(
                    f"Permission error checking PID {pid}, assuming it's running for now."
                )
                await asyncio.sleep(interval)
                elapsed += interval
                continue
            logger.error(f"Unexpected OSError checking PID {pid}: {e}")
            raise  # Other OSError, re-raise
    logger.warning(f"Process {pid} did not terminate within {timeout_seconds}s timeout.")
    return False


# --- OS Specific FFmpeg Arguments ---
def _get_os_specific_audio_input(audio_source: str) -> List[str]:
    system = platform.system()
    if audio_source == "system":
        if system == "Darwin":
            return ["-f", "avfoundation", "-i", ":1"]  # May need specific device index
        if system == "Windows":
            return [
                "-f",
                "dshow",
                "-i",
                "audio=virtual-audio-capturer",
            ]  # Requires VB-Audio Virtual Cable or similar
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        raise NotImplementedError(f"System audio capture not implemented for {system}")
    if audio_source == "microphone":
        if system == "Darwin":
            return ["-f", "avfoundation", "-i", ":0"]  # Often default mic, may need specific index
        if system == "Windows":
            return ["-f", "dshow", "-i", "audio=Microphone"]  # Generic name, might need specific
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        raise NotImplementedError(f"Microphone capture not implemented for {system}")
    raise ValueError(f"Unsupported audio_source: {audio_source}")


def _get_os_specific_video_input(video_source: str) -> List[str]:
    system = platform.system()
    if video_source == "screen":
        if system == "Darwin":
            return [
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-i",
                "1",
            ]  # Often main screen, may need specific index
        if system == "Windows":
            return ["-f", "gdigrab", "-i", "desktop"]
        if system == "Linux":
            return [
                "-f",
                "x11grab",
                "-draw_mouse",
                "1",
                "-s",
                "1920x1080",
                "-i",
                ":0.0",
            ]  # Example, adjust size/display
        raise NotImplementedError(f"Screen capture not implemented for {system}")
    if video_source == "window":
        raise NotImplementedError("Window-specific recording is not supported yet.")
    raise ValueError(f"Unsupported video_source: {video_source}")


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

    cmd: List[str] = ["ffmpeg", "-y"]  # -y overwrites output files without asking
    if record_video:
        cmd += _get_os_specific_video_input(video_source)
        cmd += [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
        ]
    if record_audio:
        cmd += _get_os_specific_audio_input(audio_source)
        cmd += ["-c:a", "aac", "-b:a", "192k"]
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
            # Attempt to delete potentially empty stderr log if ffmpeg failed to start
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

    # Add to state file
    all_states = await _read_state_file()
    print(f"Adding to state file: {rec_id} = {stored_info}")
    print(f"All states: {all_states}")
    all_states[rec_id] = stored_info
    print(f"All states after adding: {all_states}")
    await _write_state_file(all_states)

    # The stderr_file_handle is managed by the OS and FFmpeg process.
    # It will be closed when FFmpeg exits or the handle is garbage collected if this process exits.
    # We don't need to explicitly close it here if FFmpeg is running.

    logger.info(
        f"Recording started with ID {rec_id}, PID {process.pid}. "
        f"Temp: {temp_output}, Output: {final_output}"
    )
    return rec_id


async def stop_recording_tool(recording_id: str) -> str:
    all_states = await _read_state_file()
    print(f"Stopping recording {recording_id}")
    print(f"All states: {all_states}")
    info = all_states.pop(recording_id, None)

    if info is None:
        # Check if it was already stopped and removed by a concurrent call
        # If we write the state file before raising, a rapid second call might also find it missing.
        # It's safer to write the state *after* successful processing or definite failure.
        raise ToolError(f"No active recording with ID '{recording_id}' found in state file.")

    # Immediately update state file to reflect removal attempt
    await _write_state_file(all_states)

    pid = info["pid"]
    temp_output_path = Path(info["temp_output_path"])
    final_output_path = Path(info["output_path"])
    stderr_log_path = Path(info["stderr_log_path"])

    logger.info(f"Attempting to stop recording ID {recording_id} (PID: {pid})")

    # Attempt to stop FFmpeg process using signals
    terminated_gracefully = False
    try:
        if not await _wait_for_pid(pid, timeout_seconds=0.1):  # Check if already exited
            logger.info(f"Sending SIGINT to PID {pid} for recording {recording_id}")
            os.kill(pid, signal.SIGINT)
            terminated_gracefully = await _wait_for_pid(
                pid, timeout_seconds=7.0
            )  # Increased timeout for graceful shutdown

            if not terminated_gracefully:
                logger.warning(
                    f"PID {pid} (rec: {recording_id}) did not stop with SIGINT. Sending SIGTERM."
                )
                os.kill(pid, signal.SIGTERM)
                terminated_gracefully = await _wait_for_pid(pid, timeout_seconds=3.0)

                if not terminated_gracefully:
                    logger.error(
                        f"PID {pid} (rec: {recording_id}) did not stop with SIGTERM. "
                        "Sending SIGKILL."
                    )
                    os.kill(pid, signal.SIGKILL)
                    terminated_gracefully = await _wait_for_pid(
                        pid, timeout_seconds=1.0
                    )  # SIGKILL should be fast
                    if not terminated_gracefully:
                        logger.error(
                            f"PID {pid} (rec: {recording_id}) failed to stop even with SIGKILL."
                        )
        else:
            logger.info(
                f"Process PID {pid} (rec: {recording_id}) already exited before stop signals."
            )
            terminated_gracefully = True  # It's terminated, that's what matters

    except ProcessLookupError:  # os.kill can raise this if PID doesn't exist
        logger.info(f"Process PID {pid} (rec: {recording_id}) not found. Already terminated.")
        terminated_gracefully = True
    except Exception as e:
        logger.exception(
            f"Error during FFmpeg process stop for PID {pid} (rec: {recording_id}): {e}"
        )
        # Continue to file checks, as FFmpeg might have exited/crashed

    logger.info(
        f"FFmpeg process PID {pid} (rec: {recording_id}) termination status: "
        f"{'Graceful/Confirmed' if terminated_gracefully else 'Uncertain/Forced'}"
    )

    # Read and log FFmpeg's stderr
    ffmpeg_stderr_content = ""
    if stderr_log_path.exists():
        try:
            with open(stderr_log_path, "r") as f_err:
                ffmpeg_stderr_content = f_err.read()
            if ffmpeg_stderr_content:
                logger.info(
                    f"FFmpeg stderr for {recording_id} (PID: {pid}):\n{ffmpeg_stderr_content}"
                )
            else:
                logger.info(f"FFmpeg stderr log for {recording_id} (PID: {pid}) was empty.")
        except Exception as e:
            logger.warning(f"Could not read FFmpeg stderr log {stderr_log_path}: {e}")

    # Ensure temporary file exists, wait briefly if necessary
    # This loop is crucial if FFmpeg takes a moment to flush file after termination signal
    temp_file_found = False
    for attempt in range(10):  # Wait up to 10 * 0.2 = 2 seconds
        if temp_output_path.exists() and temp_output_path.stat().st_size > 0:
            temp_file_found = True
            break
        logger.debug(
            "Waiting for temp file %s (attempt %d), current size: %s",
            temp_output_path,
            attempt + 1,
            temp_output_path.stat().st_size if temp_output_path.exists() else "N/A",
        )
        await asyncio.sleep(0.2)

    if not temp_file_found:
        error_msg = (
            f"Temporary file not found or empty: {temp_output_path} for recording {recording_id}."
        )
        if ffmpeg_stderr_content:
            error_msg += f"\nFFmpeg stderr indicated: {ffmpeg_stderr_content.strip()[:500]}..."
        logger.error(error_msg)
        # Cleanup stderr log even on failure
        if stderr_log_path.exists():
            try:
                stderr_log_path.unlink()
            except Exception:
                logger.warning(f"Could not delete stderr log {stderr_log_path} on error path.")
        raise ToolError(error_msg)

    # Move to final location
    try:
        try:
            # Ensure parent directory of final_output_path exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_output_path), str(final_output_path))
        except OSError as exc:  # shutil.move can raise this, not temp_output_path.replace
            if exc.errno == errno.EXDEV:  # Cross-device link
                logger.warning(
                    f"Cross-device move for {recording_id}. Using shutil.move as fallback."
                )
                shutil.move(str(temp_output_path), str(final_output_path))
            else:
                raise
        logger.info(f"Recording {recording_id} saved to {final_output_path}")
    except Exception as exc:
        logger.exception(
            f"Failed to finalize recording {recording_id} from "
            f"{temp_output_path} to {final_output_path}"
        )
        # Attempt to clean up temp file if move failed
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                logger.warning(
                    f"Could not delete temp file {temp_output_path} after failed finalization."
                )
        # Cleanup stderr log
        if stderr_log_path.exists():
            try:
                stderr_log_path.unlink()
            except Exception:
                logger.warning(f"Could not delete stderr log {stderr_log_path} on error path.")
        raise ToolError(f"Failed to finalize recording {recording_id}: {exc}") from exc

    # Cleanup temporary files (stderr log; temp_output should be gone if move succeeded)
    try:
        if temp_output_path.exists():  # Should not exist if move was successful
            logger.warning(
                f"Temp file {temp_output_path} still exists after successful move. Deleting."
            )
            temp_output_path.unlink()
        if stderr_log_path.exists():
            stderr_log_path.unlink()

        # Attempt to remove the base temp directory if it's empty
        # This is a bit aggressive; be careful if other tools use TEMP_DIR_BASE directly
        # For now, only remove the specific recording's parent if it was rec_id based.
        # The current TEMP_DIR_BASE is shared, so don't remove it.
        # If temp_output_path.parent was unique per recording, we could rmdir it.
        # Example: if temp_output_path was TEMP_DIR_BASE / rec_id / "temp.mp4"
        # then temp_output_path.parent.rmdir() if empty.
        # For now, individual file cleanup is sufficient.

    except Exception:
        logger.warning(f"Cleanup of temp files for {recording_id} failed.", exc_info=True)

    return f"Recording saved to: {final_output_path}"
