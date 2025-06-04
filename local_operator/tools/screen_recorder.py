import asyncio
import os
import platform
import shutil
import subprocess
import uuid
from typing import Any, Dict, List

# Global dictionary to keep track of active recording processes
# Key: recording_id (str), Value: Dict containing 'ffmpeg_process',
# 'output_path', 'temp_output_path', 'record_video', 'record_audio'
_active_recordings: Dict[str, Dict[str, Any]] = {}


# Custom exception for tool-specific errors
class ToolError(Exception):
    """Custom exception for tool-related errors."""

    pass


async def _run_ffmpeg_command(command: List[str], process_name: str) -> asyncio.subprocess.Process:
    """Helper to run an FFmpeg command in a subprocess."""
    print(f"Starting {process_name} FFmpeg process with command: {' '.join(command)}")
    process = await asyncio.create_subprocess_exec(
        *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Optionally, read stderr in a separate task to prevent buffer overflow
    # asyncio.create_task(_log_stderr(process, process_name))
    return process


async def _log_stderr(process: asyncio.subprocess.Process, name: str):
    """Logs stderr from a subprocess."""
    while True:
        if process.stderr is None:
            break
        line = await process.stderr.readline()
        if not line:
            break
        print(f"[{name} STDERR] {line.decode().strip()}")


def _get_os_specific_audio_input(audio_source: str) -> List[str]:
    """Determines OS-specific FFmpeg audio input arguments."""
    system = platform.system()
    if audio_source == "system":
        if system == "Darwin":
            # Requires BlackHole or similar virtual audio device
            # User must select BlackHole as default output and input
            # in Audio MIDI Setup
            # Or specify device by name if known: -i "BlackHole 2ch"
            # :1 typically refers to BlackHole if configured as default
            return ["-f", "avfoundation", "-i", ":1"]
        elif system == "Windows":
            # Requires VB-Audio Cable or similar virtual audio device
            # User must set "CABLE Output" as default playback device
            # And "CABLE Input" as default recording device
            # Placeholder, actual device name needed
            return ["-f", "dshow", "-i", "audio=virtual-audio-capturer"]
        elif system == "Linux":
            # Uses PulseAudio or PipeWire
            # User might need to select source via portal or configure
            # 'default' usually works for system output
            return ["-f", "pulse", "-i", "default"]
        else:
            raise NotImplementedError(f"System audio capture not implemented for {system}")
    elif audio_source == "microphone":
        if system == "Darwin":
            # :0 typically refers to default microphone
            return ["-f", "avfoundation", "-i", "none:0"]
        elif system == "Windows":
            # Placeholder, actual device name needed
            return ["-f", "dshow", "-i", "audio=Microphone"]
        elif system == "Linux":
            # 'default' usually works for default microphone
            return ["-f", "pulse", "-i", "default"]
        else:
            raise NotImplementedError(f"Microphone capture not implemented for {system}")
    else:
        raise ValueError(f"Unsupported audio_source: {audio_source}")


def _get_os_specific_video_input(video_source: str) -> List[str]:
    """Determines OS-specific FFmpeg video input arguments."""
    system = platform.system()
    if video_source == "screen":
        if system == "Darwin":
            # Uses AVFoundation for screen capture
            # '1' typically refers to screen 0
            return ["-f", "avfoundation", "-i", "1"]
        elif system == "Windows":
            # Uses GDI or Desktop Duplication API (FFmpeg dshow)
            # 'desktop' captures entire screen
            return ["-f", "gdigrab", "-i", "desktop"]
        elif system == "Linux":
            # Uses x11grab for X11, or xdg-desktop-portal for Wayland
            # For simplicity, assuming X11 for direct FFmpeg or relying
            # on xdg-desktop-portal for Wayland
            # Adjust resolution and display
            return ["-f", "x11grab", "-s", "1920x1080", "-i", ":0.0"]
        else:
            raise NotImplementedError(f"Screen capture not implemented for {system}")
    elif video_source == "window":
        # This is a future enhancement, will require more complex FFmpeg
        raise NotImplementedError("Window-specific recording is a future enhancement.")
    else:
        raise ValueError(f"Unsupported video_source: {video_source}")


# --- Tool Implementations ---


async def start_recording_tool(
    output_path: str,
    record_audio: bool = True,
    record_video: bool = True,
    audio_source: str = "system",
    video_source: str = "screen",
) -> str:
    """
    Starts screen and/or audio recording.
    Returns a recording ID.
    """
    if not shutil.which("ffmpeg"):
        raise ToolError(
            "FFmpeg is required for recording. Please install it via your "
            "package manager (e.g., `brew install ffmpeg` on macOS, "
            "`apt-get install ffmpeg` on Linux, or download from "
            "ffmpeg.org on Windows)."
        )

    recording_id = str(uuid.uuid4())
    temp_dir = os.path.join(os.path.expanduser("~"), ".local_operator_recordings_temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Use .mp4 for combined output
    temp_output_path = os.path.join(temp_dir, f"{recording_id}_temp.mp4")

    # Overwrite output files without asking
    ffmpeg_command = ["ffmpeg", "-y"]

    if record_video:
        video_input_args = _get_os_specific_video_input(video_source)
        ffmpeg_command.extend(video_input_args)
        # Video encoding options
        ffmpeg_command.extend(["-r", "30", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23"])

    if record_audio:
        audio_input_args = _get_os_specific_audio_input(audio_source)
        ffmpeg_command.extend(audio_input_args)
        # Audio encoding options
        ffmpeg_command.extend(["-c:a", "aac", "-b:a", "192k"])

    if not record_video and not record_audio:
        raise ValueError("At least one of record_video or record_audio must be True.")

    ffmpeg_command.append(temp_output_path)

    process_name = (
        "Combined Recorder"
        if record_video and record_audio
        else ("Video Recorder" if record_video else "Audio Recorder")
    )
    ffmpeg_process = await _run_ffmpeg_command(ffmpeg_command, process_name)

    _active_recordings[recording_id] = {
        "ffmpeg_process": ffmpeg_process,
        "output_path": output_path,
        "temp_output_path": temp_output_path,
        "record_video": record_video,
        "record_audio": record_audio,
    }

    return recording_id


async def stop_recording_tool(recording_id: str) -> str:
    """
    Stops an active recording and finalizes the output file.
    """
    if recording_id not in _active_recordings:
        raise ValueError(f"No active recording found with ID: {recording_id}")

    recording_info = _active_recordings[recording_id]
    ffmpeg_process = recording_info["ffmpeg_process"]
    output_path = recording_info["output_path"]
    temp_output_path = recording_info["temp_output_path"]
    record_video = recording_info["record_video"]
    record_audio = recording_info["record_audio"]

    # Terminate recording process
    if ffmpeg_process and ffmpeg_process.returncode is None:
        ffmpeg_process.terminate()
        await ffmpeg_process.wait()
        print(f"FFmpeg process for {recording_id} terminated.")

    # If both video and audio were recorded together, or only one was recorded,
    # the output is already in temp_output_path. Just move it.
    if record_video or record_audio:
        os.rename(temp_output_path, output_path)
        print(f"Recording saved to {output_path}")
    else:
        raise ValueError("Neither video nor audio was set to be recorded for this " "recording ID.")

    # Clean up temporary files
    # The PRD had temp_video_path and temp_audio_path which are not defined
    # in this context.
    # Assuming temp_output_path is the only temporary file to clean up.
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)

    temp_dir = os.path.dirname(temp_output_path)
    # Remove temp dir if empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)

    del _active_recordings[recording_id]
    return f"Recording saved to: {output_path}"


# --- Tool Registration (Conceptual, would be in ToolRegistry.init_tools) ---
# self.add_tool("start_recording", start_recording_tool)
# self.add_tool("stop_recording", stop_recording_tool)
