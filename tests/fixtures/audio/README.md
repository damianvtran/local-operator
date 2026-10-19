# Waveform fixture

`tone.mp3` is a synthetic 440 Hz sine, 0.12 seconds of source audio at 8 kHz,
encoded as 24 kbps MP3 (1,125 bytes). It contains no recorded/user audio.
Generated with FFmpeg; tests use the committed bytes and need no FFmpeg binary:

```sh
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i 'sine=frequency=440:duration=0.12:sample_rate=8000' \
  -c:a libmp3lame -b:a 24k -map_metadata -1 tone.mp3
```

The real MP3 fixture distinguishes installed decoder availability from a mocked
sidecar writer. Codec padding can make decoded duration longer than the source.
