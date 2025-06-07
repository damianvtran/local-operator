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
async def test_start_recording_with_system_audio_on_mac(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with (
        patch("platform.system", return_value="Darwin"),
        patch(
            "local_operator.tools.screen_recorder._find_avfoundation_device_index"
        ) as mock_find_device,
    ):
        mock_find_device.side_effect = [
            1,
            2,
        ]
        await start_recording_tool("test.mp4", audio_source="system")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "1:2" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_with_microphone_audio_on_mac(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with (
        patch("platform.system", return_value="Darwin"),
        patch(
            "local_operator.tools.screen_recorder._find_avfoundation_device_index"
        ) as mock_find_device,
    ):
        mock_find_device.side_effect = [
            1,
            0,
        ]
        await start_recording_tool("test.mp4", audio_source="microphone")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "1:0" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_with_system_audio_on_windows(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Windows"):
        await start_recording_tool("test.mp4", audio_source="system")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "audio=virtual-audio-capturer" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_with_microphone_audio_on_windows(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Windows"):
        await start_recording_tool("test.mp4", audio_source="microphone")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "audio=Microphone" in call_args


@pytest.mark.asyncio
@patch("local_operator.tools.screen_recorder.shutil.which", return_value=True)
@patch("local_operator.tools.screen_recorder._run_ffmpeg_command", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder._read_state_file", return_value={})
@patch("local_operator.tools.screen_recorder._write_state_file", new_callable=AsyncMock)
@patch("local_operator.tools.screen_recorder.Path.exists", return_value=True)
@patch("local_operator.tools.screen_recorder.Path.read_text", return_value="frame=1")
async def test_start_recording_with_system_audio_on_linux(
    mock_read_text,
    mock_path_exists,
    mock_write_state,
    mock_read_state,
    mock_run_ffmpeg,
    mock_which,
):
    with patch("platform.system", return_value="Linux"):
        await start_recording_tool("test.mp4", audio_source="system")
        mock_run_ffmpeg.assert_called_once()
        call_args = mock_run_ffmpeg.call_args[0][0]
        assert "default" in call_args
