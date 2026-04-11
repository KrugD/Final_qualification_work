"""Assemble Stage 2 dataset from pipeline_output + audio_chunks.

Reads summarization results from pipeline_output/<chunk_name>/,
finds matching audio from audio_chunks/<source>/<part>.wav,
and creates data/protocols/<sample_NNN>/{audio.wav, protocol.txt}.

Usage:
    python scripts/prepare_stage2_data.py
    python scripts/prepare_stage2_data.py --symlink   # symlink audio instead of copying
"""

import argparse
import os
import re
import shutil
from pathlib import Path


PIPELINE_OUTPUT = Path("pipeline_output")
AUDIO_CHUNKS = Path("audio_chunks")
OUTPUT_DIR = Path("data/protocols")


def parse_summarization(path: Path) -> str | None:
    """Parse pipeline summarization output into clean protocol text.

    Expected format per speaker block:
        Speaker: SPEAKER_XX
        ...
        Summary: <text>
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("=" * 50)

    speakers = []
    for block in blocks:
        block = block.strip()
        if not block or "TEXT SUMMARIZATION RESULTS" in block:
            continue

        speaker_match = re.search(r"Speaker:\s*(\S+)", block)
        summary_match = re.search(r"Summary:\s*(.+)", block, re.DOTALL)

        if speaker_match and summary_match:
            speaker = speaker_match.group(1)
            summary = summary_match.group(1).strip()
            if summary:
                speakers.append(f"{speaker}: {summary}")

    if not speakers:
        return None

    return "\n\n".join(speakers)


def find_audio_path(chunk_name: str) -> Path | None:
    """Map pipeline_output folder name back to audio_chunks WAV file.

    Pipeline output:  audio_2018_88m41s_part_00
    Audio chunks:     audio_chunks/audio_2018_88m41s/part_00.wav
    """
    match = re.match(r"^(.+)_(part_\d+)$", chunk_name)
    if not match:
        return None

    source = match.group(1)
    part = match.group(2)

    wav_path = AUDIO_CHUNKS / source / f"{part}.wav"
    if wav_path.exists():
        return wav_path

    return None


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 2 dataset")
    parser.add_argument("--symlink", action="store_true",
                        help="Create symlinks to audio instead of copying")
    args = parser.parse_args()

    if not PIPELINE_OUTPUT.exists():
        print(f"Error: {PIPELINE_OUTPUT} not found")
        return
    if not AUDIO_CHUNKS.exists():
        print(f"Error: {AUDIO_CHUNKS} not found")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_dirs = sorted(
        d for d in PIPELINE_OUTPUT.iterdir()
        if d.is_dir()
    )

    print(f"Found {len(chunk_dirs)} pipeline output folders")

    created = 0
    skipped_no_summ = 0
    skipped_no_audio = 0
    skipped_empty = 0

    for chunk_dir in chunk_dirs:
        chunk_name = chunk_dir.name

        summ_files = list(chunk_dir.glob("*_summarization.txt"))
        if not summ_files:
            skipped_no_summ += 1
            continue

        protocol_text = parse_summarization(summ_files[0])
        if not protocol_text:
            skipped_empty += 1
            continue

        audio_path = find_audio_path(chunk_name)
        if audio_path is None:
            skipped_no_audio += 1
            continue

        sample_dir = OUTPUT_DIR / f"sample_{created + 1:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        (sample_dir / "protocol.txt").write_text(protocol_text, encoding="utf-8")

        target_audio = sample_dir / "audio.wav"
        if args.symlink:
            if target_audio.exists() or target_audio.is_symlink():
                target_audio.unlink()
            target_audio.symlink_to(audio_path.resolve())
        else:
            shutil.copy2(str(audio_path), str(target_audio))

        created += 1

    print(f"\nDone!")
    print(f"  Created:          {created} samples")
    print(f"  Skipped (no summ): {skipped_no_summ}")
    print(f"  Skipped (no audio): {skipped_no_audio}")
    print(f"  Skipped (empty):   {skipped_empty}")
    print(f"  Output:           {OUTPUT_DIR}/")

    if created > 0:
        example = OUTPUT_DIR / "sample_0001"
        print(f"\nExample: {example}")
        print(f"  audio.wav:    {os.path.getsize(example / 'audio.wav') / 1e6:.1f} MB")
        protocol = (example / "protocol.txt").read_text(encoding="utf-8")
        print(f"  protocol.txt: {len(protocol)} chars")
        print(f"  Preview:      {protocol[:200]}...")


if __name__ == "__main__":
    main()
