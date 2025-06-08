from unittest.mock import AsyncMock, patch

import pytest

from local_operator.tools.screen_recorder import start_recording_tool


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_on_mac(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Darwin"):
        await start_recording_tool(
            "test.mp4", video_device="screen-capture-device", audio_device="mic-device"
        )
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "screen-capture-device:mic-device" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_on_windows_desktop(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Windows"):
        await start_recording_tool(
            "test.mp4", video_device="desktop", audio_device="virtual-audio-capturer"
        )
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "gdigrab" in call_args
        assert "desktop" in call_args
        assert "audio=virtual-audio-capturer" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_on_windows_webcam(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Windows"):
        await start_recording_tool(
            "test.mp4", video_device="Integrated Webcam", audio_device="Microphone"
        )
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "video=Integrated Webcam" in call_args
        assert "audio=Microphone" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_on_linux(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Linux"):
        await start_recording_tool("test.mp4", video_device=":0.0", audio_device="default")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "x11grab" in call_args
        assert ":0.0" in call_args
        assert "pulse" in call_args
        assert "default" in call_args
