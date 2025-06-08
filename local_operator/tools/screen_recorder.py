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
import time
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
        logger.info(f"Spawned FFmpeg process with PID: {process.pid} for {process_name}")
        # Give it a moment to potentially fail and check its status
        await asyncio.sleep(1.0)
        if process.returncode is not None:
            logger.error(
                f"FFmpeg process {process.pid} for {process_name} "
                f"terminated immediately with code {process.returncode}."
            )
            # The calling function will handle the cleanup and error reporting
            # based on the process having exited.
    except Exception as exc:
        logger.exception("Failed to start FFmpeg process %s", process_name)
        if stderr_handle:
            try:
                stderr_handle.close()
            except Exception:
                pass
        # Be more specific about the error
        error_message = (
            f"Could not start FFmpeg process '{process_name}'. Reason: {type(exc).__name__}: {exc}"
        )
        raise ToolError(error_message) from exc
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


async def start_recording_tool(
    output_path: str,
    record_audio: bool = True,
    record_video: bool = True,
    audio_device: str | None = None,
    video_device: str | None = None,
) -> str:
    """Start recording screen and/or audio using FFmpeg.

    This tool allows you to start recording from a specific video and/or audio device.
    The recording will continue until you call the stop_recording_tool. You must provide
    an output path where the recording will be saved.  Make an appropriate
    and well-organized path and file name at your discretion unless the user
    specifically tells you where to save the file.

    To use this tool, you must first know the names of the devices you want to record from.
    You can list the available devices by running the appropriate FFmpeg command in the
    terminal. For example, on macOS, you can list devices with:
    `ffmpeg -f avfoundation -list_devices true -i ""`.  Do not request
    the user to run this command.  Instead, you must first run the command
    and then use the output to determine the device names.

    You must provide the exact device name for the `video_device` and/or `audio_device`
    arguments.

    Args:
        output_path (str): Path where the recording file will be saved (e.g., "recording.mp4").
            Pick a well-organized path and file name at your discretion unless the user
            specifically tells you where to save the file.
        record_audio (bool, optional): Whether to record audio. Defaults to True. If True,
            `audio_device` must be provided.
        record_video (bool, optional): Whether to record video. Defaults to True. If True,
            `video_device` must be provided.
        audio_device (str, optional): The name of the audio device to record from.
        video_device (str, optional): The name of the video device to record from. This can be
            a screen capture device or a webcam.

    Returns:
        str: The unique ID for the recording session.

    Raises:
        ToolError: If FFmpeg is not installed or not found in PATH.
        ValueError: If recording is enabled for a device type but no device name is provided,
            or if neither audio nor video recording is enabled.
    """
    if shutil.which("ffmpeg") is None:
        raise ToolError("FFmpeg executable not found. Please install FFmpeg.")
    if not (record_audio or record_video):
        raise ValueError("At least one of record_audio or record_video must be True.")
    if record_video and not video_device:
        raise ValueError("video_device must be provided when record_video is True.")
    if record_audio and not audio_device:
        raise ValueError("audio_device must be provided when record_audio is True.")

    rec_id = str(uuid.uuid4())
    TEMP_DIR_BASE.mkdir(parents=True, exist_ok=True)

    temp_output = TEMP_DIR_BASE / f"{rec_id}_temp.mp4"
    stderr_log_path = TEMP_DIR_BASE / f"{rec_id}_stderr.log"
    final_output = Path(output_path).expanduser().resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)

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

    cmd: List[str] = ["ffmpeg", "-y"]
    system = platform.system()

    # Input configuration
    if system == "Darwin":
        input_spec = ""
        if record_video:
            assert video_device is not None
            input_spec += video_device
        if record_audio:
            assert audio_device is not None
            input_spec = f":{audio_device}" if not record_video else f"{input_spec}:{audio_device}"

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
            input_spec,
        ]

    elif system == "Windows":
        if record_video:
            assert video_device is not None
            if video_device == "desktop":
                cmd += ["-f", "gdigrab", "-framerate", "30", "-i", "desktop"]
            else:
                cmd += ["-f", "dshow", "-i", f"video={video_device}"]
        if record_audio:
            assert audio_device is not None
            cmd += ["-f", "dshow", "-i", f"audio={audio_device}"]

    elif system == "Linux":
        if record_video:
            assert video_device is not None
            if ":" in video_device:
                cmd += [
                    "-f",
                    "x11grab",
                    "-draw_mouse",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    video_device,
                ]
            else:
                cmd += ["-f", "v4l2", "-framerate", "30", "-i", video_device]
        if record_audio:
            assert audio_device is not None
            cmd += ["-f", "pulse", "-i", audio_device]
    else:
        raise NotImplementedError(f"Recording not implemented for {system}")

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
    process = None
    try:
        stderr_file_handle = open(stderr_log_path, "wb")
        process = await _run_ffmpeg_command(cmd, proc_name, stderr_handle=stderr_file_handle)

        # Wait for ffmpeg to confirm recording start or initial file output
        startup_timeout = 5.0
        success_pattern = re.compile(r"Press \[q\] to stop|frame=\s*\d+|size=\s*\d+")
        start_time = time.monotonic()
        error_output = ""
        started = False

        while True:
            # Check if the temp file has started receiving data
            try:
                if temp_output.exists() and temp_output.stat().st_size > 0:
                    started = True
                    break
            except Exception:
                pass

            # Read available stderr content for progress or errors
            if stderr_log_path.exists():
                try:
                    error_output = stderr_log_path.read_text(errors="ignore")
                except Exception:
                    error_output = ""
                if success_pattern.search(error_output):
                    started = True
                    break
                # Detect obvious errors
                if re.search(
                    r"[Ee]rror|not authorized|permission denied", error_output, re.IGNORECASE
                ):
                    break

            # Check if process exited prematurely
            if process.returncode is not None:
                break

            # Timeout
            if time.monotonic() - start_time > startup_timeout:
                break

            await asyncio.sleep(0.1)

        if not started:
            # Cleanup ffmpeg process
            if process.returncode is None:
                try:
                    process.kill()
                except Exception:
                    pass

            stderr_excerpt = error_output.strip()
            raise ToolError(
                f"Failed to start recording (timeout or no data detected). "
                f"FFmpeg stderr (excerpt):\n{stderr_excerpt}"
            )

        # On successful start, record state
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

    except Exception:
        # Cleanup on failure
        if stderr_file_handle:
            try:
                stderr_file_handle.close()
            except Exception:
                pass
        if stderr_log_path.exists():
            try:
                stderr_log_path.unlink()
            except Exception:
                logger.warning(f"Could not delete stderr log {stderr_log_path}")
        raise


async def stop_recording_tool(recording_id: str) -> str:
    """Stop an active screen/audio recording and save the final output file.

    This tool stops a recording that was previously started with start_recording_tool.
    It gracefully terminates the FFmpeg process, waits for the recording to be finalized,
    and moves the temporary recording file to the specified output location. The tool
    will attempt graceful termination first, then escalate to forceful termination if
    necessary.

    Args:
        recording_id (str): The unique identifier returned by start_recording_tool
            when the recording was started.

    Returns:
        str: Confirmation message indicating the recording was stopped and the location
        of the saved file.

    Raises:
        ToolError: If no active recording with the given ID is found, if the recording
            process cannot be terminated, or if the output file cannot be saved.
    """
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

    # Wait for temp file flush (increased attempts for robustness)
    temp_file_found = False
    max_attempts = 15
    for _ in range(max_attempts):
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
