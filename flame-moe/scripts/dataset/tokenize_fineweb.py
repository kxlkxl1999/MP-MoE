#!/usr/bin/env python3
"""Tokenize FineWeb-Edu dataset using Megatron-LM's preprocess_data.py."""

import argparse
import multiprocessing
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize FineWeb-Edu dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/pythia-12b")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--megatron-path", type=str, default="./Megatron-LM")
    parser.add_argument("--append-eod", action="store_true", default=True)
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--delete-after-tokenize", action="store_true",
                        help="Delete JSONL after tokenizing to save space")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already tokenized files")
    return parser.parse_args()


def tokenize_file(input_file, output_prefix, tokenizer, megatron_path, workers, append_eod):
    """Tokenize a single JSONL file using Megatron-LM's preprocess_data.py."""
    preprocess_script = Path(megatron_path) / "tools" / "preprocess_data.py"

    cmd = [
        sys.executable,
        str(preprocess_script),
        "--input", str(input_file),
        "--output-prefix", str(output_prefix),
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", tokenizer,
        "--workers", str(workers),
    ]

    if append_eod:
        cmd.append("--append-eod")

    print(f"Tokenizing: {input_file.name}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError:
        return False


def is_tokenized(output_prefix):
    """Check if file is already tokenized."""
    bin_file = Path(f"{output_prefix}_text_document.bin")
    idx_file = Path(f"{output_prefix}_text_document.idx")
    return bin_file.exists() and idx_file.exists()


def main():
    args = parse_args()

    if args.workers is None:
        args.workers = multiprocessing.cpu_count()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_dir)
    if args.input_file:
        jsonl_files = [Path(args.input_file)]
    else:
        jsonl_files = sorted([f for f in input_path.glob("*.jsonl") if f.name != "sample.jsonl"])

    if not jsonl_files:
        print(f"Error: No JSONL files in {input_path}")
        return 1

    print(f"Found {len(jsonl_files)} file(s) to tokenize")

    success_count = 0
    skipped_count = 0
    deleted_size = 0

    for jsonl_file in jsonl_files:
        output_prefix = output_path / jsonl_file.stem

        # Resume: skip already tokenized
        if args.resume and is_tokenized(output_prefix):
            print(f"Skipping (already tokenized): {jsonl_file.name}")
            skipped_count += 1
            continue

        if tokenize_file(jsonl_file, output_prefix, args.tokenizer,
                        args.megatron_path, args.workers, args.append_eod):
            success_count += 1

            bin_file = Path(f"{output_prefix}_text_document.bin")
            if bin_file.exists():
                size_gb = bin_file.stat().st_size / (1024**3)
                print(f"  Output: {size_gb:.2f} GB")

            # Delete source to save space
            if args.delete_after_tokenize:
                file_size = jsonl_file.stat().st_size
                jsonl_file.unlink()
                deleted_size += file_size
                print(f"  Deleted: {jsonl_file.name} (freed {file_size / (1024**3):.2f} GB)")

    print(f"\nComplete: {success_count} tokenized, {skipped_count} skipped")
    if deleted_size > 0:
        print(f"Total space freed: {deleted_size / (1024**3):.2f} GB")
    return 0 if success_count + skipped_count == len(jsonl_files) else 1


if __name__ == "__main__":
    exit(main())
