"""Split long audio files into ~7 minute chunks for pipeline processing.

Uses ffmpeg segment muxer for fast, memory-efficient splitting.
Reads from audio_test/, writes to audio_chunks/.
Original files are NOT modified.

Usage:
    python scripts/split_audio.py
"""

import json
import subprocess
import sys
from pathlib import Path

INPUT_DIR = Path("audio_test")
OUTPUT_DIR = Path("audio_chunks")
SEGMENT_SEC = 420  # 7 minutes


def get_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def split_file(audio_path: Path, output_base: Path) -> int:
    stem = audio_path.stem
    duration = get_duration(audio_path)
    duration_min = duration / 60

    out_dir = output_base / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    if duration <= SEGMENT_SEC + 60:
        out_path = out_dir / "part_01.wav"
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y",
                "-i", str(audio_path),
                "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(out_path),
            ],
            check=True,
        )
        print(f"  {stem}: {duration_min:.1f} min -> 1 chunk (no split needed)")
        return 1

    pattern = str(out_dir / "part_%02d.wav")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-i", str(audio_path),
            "-f", "segment",
            "-segment_time", str(SEGMENT_SEC),
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            pattern,
        ],
        check=True,
    )

    chunks = sorted(out_dir.glob("part_*.wav"))
    num_chunks = len(chunks)

    durations = []
    for ch in chunks:
        d = get_duration(ch) / 60
        durations.append(f"{d:.1f}")

    print(f"  {stem}: {duration_min:.1f} min -> {num_chunks} chunks ({', '.join(durations)} min)")
    return num_chunks


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory {INPUT_DIR} not found")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
    )

    print(f"Found {len(audio_files)} audio files in {INPUT_DIR}/", flush=True)
    print(f"Output directory: {OUTPUT_DIR}/", flush=True)
    print(f"Segment size: ~{SEGMENT_SEC // 60} min", flush=True)
    print(flush=True)

    total_chunks = 0
    for i, audio_path in enumerate(audio_files, 1):
        try:
            n = split_file(audio_path, OUTPUT_DIR)
            total_chunks += n
            sys.stdout.flush()
        except Exception as e:
            print(f"  ERROR [{i}/{len(audio_files)}] {audio_path.name}: {e}", flush=True)

    print(f"\nDone! Created {total_chunks} chunks from {len(audio_files)} files.", flush=True)
    print(f"Output: {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
