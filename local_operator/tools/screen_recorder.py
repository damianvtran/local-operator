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
async def _find_avfoundation_device_index(device_type: str, hint: str) -> int | None:
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
        return None

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
    return found_index


# --- OS Specific FFmpeg Arguments ---
async def _get_os_specific_video_input(video_source: str) -> List[str]:
    system = platform.system()
    if video_source == "screen":
        if system == "Darwin":
            idx = await _find_avfoundation_device_index("video", "capture screen")
            if idx is None:
                logger.warning("Could not find screen capture device, using default index 1.")
                idx = 1
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

    if system == "Darwin":
        if audio_source == "system":
            # On macOS, system audio capture usually requires a virtual audio device.
            # We'll try to find one, otherwise fall back to the microphone.
            idx = await _find_avfoundation_device_index("audio", "blackhole")
            if idx is not None:
                logger.info("Found system audio device (BlackHole).")
                return ["-f", "avfoundation", "-thread_queue_size", "512", "-i", f":{idx}"]

            logger.warning(
                "Could not find a 'BlackHole' system audio device. "
                "Falling back to default microphone input. "
                "Please install BlackHole for system audio recording on macOS."
            )
            # Fall through to microphone
            audio_source = "microphone"

        if audio_source == "microphone":
            idx = await _find_avfoundation_device_index("audio", "microphone")
            if idx is None:
                logger.warning("Could not find microphone, using default index 0.")
                idx = 0
            return ["-f", "avfoundation", "-thread_queue_size", "512", "-i", f":{idx}"]

    elif system == "Windows":
        if audio_source == "system":
            return ["-f", "dshow", "-i", "audio=virtual-audio-capturer"]
        if audio_source == "microphone":
            return ["-f", "dshow", "-i", "audio=Microphone"]

    elif system == "Linux":
        if audio_source in ["system", "microphone"]:
            return ["-f", "pulse", "-i", "default"]

    raise ValueError(f"Unsupported audio_source '{audio_source}' for system '{system}'")


# --- Public Tool Functions ---
async def start_recording_tool(
    output_path: str,
    record_audio: bool = True,
    record_video: bool = True,
    audio_source: str = "system",
    video_source: str = "screen",
) -> str:
    """Start recording screen and/or audio using FFmpeg.

    This tool allows you to start recording the screen, audio, or both simultaneously.
    The recording will continue until you call the stop_recording_tool. You must provide
    an output path where the recording will be saved.

    By default, this tool will attempt to capture system audio. If you need to record
    from a microphone, set the `audio_source` parameter to `'microphone'`. On macOS,
    system audio capture may require a virtual audio driver like BlackHole to be installed.

    Args:
        output_path (str): Path where the recording file will be saved (e.g., "recording.mp4").  Pick a well organized path and file name at your discretion unless the user specifically tells you where to save the file.  Don't ask them for this information if they don't specify it.
        record_audio (bool, optional): Whether to record audio. Defaults to True.
        record_video (bool, optional): Whether to record video. Defaults to True.
        audio_source (str, optional): Audio source to record from. Options are "system"
            for system audio or "microphone" for microphone input. Defaults to "system".
            Unless otherwise specified, you should use the default "system" audio source.
        video_source (str, optional): Video source to record from. Currently only
            "screen" is supported. Defaults to "screen".

    Returns:
        str: The unique ID for the recording session.

    Raises:
        ToolError: If FFmpeg is not installed or not found in PATH.
        ValueError: If neither audio nor video recording is enabled, or if invalid
            source options are provided.
        NotImplementedError: If unsupported video_source options are used.
    """  # noqa: E501
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
        if v_idx is None:
            logger.warning("Could not find screen capture device, using default index 1.")
            v_idx = 1

        a_idx = None
        if audio_source == "system":
            a_idx = await _find_avfoundation_device_index("audio", "blackhole")
            if a_idx is not None:
                logger.info("Found system audio device (BlackHole) for combined recording.")
            else:
                logger.warning(
                    "Could not find a 'BlackHole' system audio device for combined recording. "
                    "Falling back to microphone."
                )
                # Fall through to try microphone

        if a_idx is None:  # Either microphone source, or system source that fell through
            a_idx = await _find_avfoundation_device_index("audio", "microphone")
            if a_idx is None:
                logger.warning("Could not find any audio device, using default index 0.")
                a_idx = 0

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
