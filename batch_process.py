"""Batch-process all audio chunks through the pipeline.

Iterates over audio_chunks/<source>/<part_XX.wav>, runs the full pipeline
(diarization → ASR → correction → summarization) on each chunk, and saves
results into pipeline_output/<source>_<part>/.

Usage:
    python batch_process.py                        # process all chunks
    python batch_process.py --limit 50             # process first 50 chunks only
    python batch_process.py --resume               # skip already processed chunks
    python batch_process.py --skip-correction      # skip correction stage (faster)
"""

import argparse
import time
from pathlib import Path

from pipeline.pipeline import run_complete_pipeline
from utils.models import clear_model_cache


CHUNKS_DIR = Path("audio_chunks")
OUTPUT_DIR = Path("pipeline_output")


def collect_chunks():
    return sorted(
        wav
        for source_dir in sorted(CHUNKS_DIR.iterdir()) if source_dir.is_dir()
        for wav in sorted(source_dir.glob("part_*.wav"))
    )


def main():
    parser = argparse.ArgumentParser(description="Batch pipeline processing for audio chunks")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N chunks (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip chunks whose output directory already exists")
    parser.add_argument("--skip-correction", action="store_true",
                        help="Skip text correction stage (faster, model sage-m2m100 not loaded)")
    args = parser.parse_args()

    all_chunks = collect_chunks()
    if args.limit > 0:
        all_chunks = all_chunks[:args.limit]

    print(f"Chunks to process: {len(all_chunks)}")
    print(f"Output directory:  {OUTPUT_DIR}/")
    if args.skip_correction:
        print(f"Correction stage:  SKIPPED")
    print()

    ok, fail = 0, 0
    t_start = time.time()

    for i, chunk_wav in enumerate(all_chunks, 1):
        chunk_name = f"{chunk_wav.parent.name}_{chunk_wav.stem}"
        out = OUTPUT_DIR / chunk_name

        if args.resume and out.exists() and any(out.glob("*_summarization.txt")):
            print(f"[{i}/{len(all_chunks)}] SKIP (already done): {chunk_name}")
            ok += 1
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(all_chunks)}] {chunk_wav}")
        print(f"{'=' * 60}")

        try:
            success, _ = run_complete_pipeline(
                str(chunk_wav), str(out),
                skip_correction=args.skip_correction,
            )
            if success:
                ok += 1
                print(f"OK: {chunk_name}")
            else:
                fail += 1
                print(f"FAIL: {chunk_name}")
        except Exception as e:
            fail += 1
            print(f"ERROR: {chunk_name}: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed / 3600:.1f} h  |  OK: {ok}  |  FAIL: {fail}")
    print(f"{'=' * 60}")

    clear_model_cache()


if __name__ == "__main__":
    main()
