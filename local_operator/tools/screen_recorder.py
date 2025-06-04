# Start of Selection
import asyncio
import logging
import platform
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, TypedDict

logger = logging.getLogger(__name__)


class RecordingInfo(TypedDict):
    ffmpeg_process: asyncio.subprocess.Process
    output_path: Path
    temp_output_path: Path
    record_video: bool
    record_audio: bool


_active_recordings: Dict[str, RecordingInfo] = {}
_active_recordings_lock = asyncio.Lock()


class ToolError(Exception):
    """Custom exception for tool-related errors."""


async def _run_ffmpeg_command(command: List[str], process_name: str) -> asyncio.subprocess.Process:
    """
    Starts an FFmpeg subprocess asynchronously.

    Args:
        command: FFmpeg command and arguments.
        process_name: Identifier for logging.

    Returns:
        The FFmpeg subprocess.

    Raises:
        ToolError: If the subprocess cannot be started.
    """
    logger.info("Launching %s with command: %s", process_name, " ".join(command))
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception as exc:
        logger.exception("Error starting FFmpeg process %s", process_name)
        raise ToolError(f"Could not start FFmpeg process: {exc}") from exc
    return process


def _get_os_specific_audio_input(audio_source: str) -> List[str]:
    """
    Returns OS-specific FFmpeg audio input arguments.

    Args:
        audio_source: 'system' or 'microphone'.

    Raises:
        ValueError: For unsupported audio_source.
        NotImplementedError: For unsupported platforms.
    """
    system = platform.system()
    if audio_source == "system":
        if system == "Darwin":
            return ["-f", "avfoundation", "-i", ":1"]
        if system == "Windows":
            return ["-f", "dshow", "-i", "audio=virtual-audio-capturer"]
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        raise NotImplementedError(f"System audio capture not implemented for {system}")
    if audio_source == "microphone":
        if system == "Darwin":
            return ["-f", "avfoundation", "-i", "none:0"]
        if system == "Windows":
            return ["-f", "dshow", "-i", "audio=Microphone"]
        if system == "Linux":
            return ["-f", "pulse", "-i", "default"]
        raise NotImplementedError(f"Microphone capture not implemented for {system}")
    raise ValueError(f"Unsupported audio_source: {audio_source}")


def _get_os_specific_video_input(video_source: str) -> List[str]:
    """
    Returns OS-specific FFmpeg video input arguments.

    Args:
        video_source: 'screen' or 'window'.

    Raises:
        ValueError: For unsupported video_source.
        NotImplementedError: For unsupported platforms.
    """
    system = platform.system()
    if video_source == "screen":
        if system == "Darwin":
            return ["-f", "avfoundation", "-i", "1"]
        if system == "Windows":
            return ["-f", "gdigrab", "-i", "desktop"]
        if system == "Linux":
            return ["-f", "x11grab", "-s", "1920x1080", "-i", ":0.0"]
        raise NotImplementedError(f"Screen capture not implemented for {system}")
    if video_source == "window":
        raise NotImplementedError("Window-specific recording is not supported yet.")
    raise ValueError(f"Unsupported video_source: {video_source}")


async def start_recording_tool(
    output_path: str,
    record_audio: bool = True,
    record_video: bool = True,
    audio_source: str = "system",
    video_source: str = "screen",
) -> str:
    """
    Starts screen and/or audio recording.

    Args:
        output_path: Destination file path for the recording.
        record_audio: Enable audio capture.
        record_video: Enable video capture.
        audio_source: 'system' or 'microphone'.
        video_source: 'screen' or 'window'.

    Returns:
        Recording ID.

    Raises:
        ToolError: If prerequisites are not met or FFmpeg fails.
        ValueError: If both record_audio and record_video are False.
    """
    if not shutil.which("ffmpeg"):
        raise ToolError("FFmpeg is required. Install it via your package manager.")
    if not (record_audio or record_video):
        raise ValueError("At least one of record_audio or record_video must be True.")

    rec_id = str(uuid.uuid4())
    temp_dir = Path.home() / ".local_operator" / "tmp" / "recordings"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_output = temp_dir / f"{rec_id}_temp.mp4"
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = ["ffmpeg", "-y"]
    if record_video:
        cmd += _get_os_specific_video_input(video_source)
        cmd += ["-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"]
    if record_audio:
        cmd += _get_os_specific_audio_input(audio_source)
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(temp_output)]

    if record_audio and record_video:
        proc_name = "CombinedRecorder"
    elif record_video:
        proc_name = "VideoRecorder"
    else:
        proc_name = "AudioRecorder"
    process = await _run_ffmpeg_command(cmd, proc_name)

    info: RecordingInfo = {
        "ffmpeg_process": process,
        "output_path": out_path,
        "temp_output_path": temp_output,
        "record_video": record_video,
        "record_audio": record_audio,
    }
    async with _active_recordings_lock:
        _active_recordings[rec_id] = info
    return rec_id


async def stop_recording_tool(recording_id: str) -> str:
    """
    Stops an active recording and finalizes the output file.

    Args:
        recording_id: ID returned by start_recording_tool().

    Returns:
        Message with final recording path.

    Raises:
        ToolError: If no active recording or if cleanup fails.
    """
    async with _active_recordings_lock:
        info = _active_recordings.pop(recording_id, None)
    if not info:
        raise ToolError(f"No active recording with ID: {recording_id}")

    proc = info["ffmpeg_process"]
    if proc.returncode is None:
        proc.terminate()
        await proc.wait()
        logger.info("FFmpeg process %s terminated", recording_id)

    temp_output = info["temp_output_path"]
    final_output = info["output_path"]
    try:
        if not temp_output.exists():
            raise ToolError(f"Temporary file not found: {temp_output}")
        temp_output.replace(final_output)
        logger.info("Recording saved to %s", final_output)
    except Exception as exc:
        logger.exception("Failed to finalize recording")
        raise ToolError(f"Failed to finalize recording: {exc}") from exc
    finally:
        try:
            if temp_output.exists():
                temp_output.unlink()
            temp_parent = temp_output.parent
            if temp_parent.exists() and not any(temp_parent.iterdir()):
                temp_parent.rmdir()
        except Exception:
            logger.warning("Cleanup of temporary files failed", exc_info=True)

    return f"Recording saved to: {final_output}"


# End of Selectio
