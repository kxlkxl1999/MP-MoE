#!/usr/bin/env python3
"""Download FineWeb-Edu dataset from Hugging Face with chunked output."""

import os
# Set HF mirror before importing huggingface libraries
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import io
import json
import shutil
import struct
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Dataset repo ID (e.g., HuggingFaceFW/fineweb-edu)",
    )
    parser.add_argument("--subset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000,
                        help="Examples per chunk file (0 = single file)")
    parser.add_argument("--resume", action="store_true", help="Resume from last chunk")
    parser.add_argument("--tokenized-dir", type=str, default=None,
                        help="Tokenized output dir to check for completed chunks")
    return parser.parse_args()


# Estimated sample counts for FineWeb-Edu subsets
SUBSET_SIZES = {
    "sample-10BT": 9_672_101,
    "sample-100BT": 96_566_469,
    "sample-350BT": 336_499_763,
}


def validate_parquet(path: Path) -> bool:
    """Validate that a parquet file is not corrupted."""
    try:
        pq.read_schema(path)
        return True
    except Exception:
        return False


def get_completed_chunks(output_path, tokenized_dir=None):
    """Get set of completed chunk indices."""
    completed = set()
    # Check existing jsonl files
    for f in output_path.glob("chunk_*.jsonl"):
        try:
            idx = int(f.stem.split("_")[1])
            completed.add(idx)
        except (IndexError, ValueError):
            pass
    # Check tokenized bins (chunks that were processed and deleted)
    if tokenized_dir:
        tokenized_path = Path(tokenized_dir)
        for f in tokenized_path.glob("chunk_*_text_document.bin"):
            idx_file = f.with_suffix(".idx")
            if not idx_file.exists():
                continue  # Incomplete tokenization, need to re-download
            try:
                idx = int(f.stem.split("_")[1])
                completed.add(idx)
            except (IndexError, ValueError):
                pass
    return completed


def load_processed_parquets(checkpoint_file: Path) -> tuple[set, int]:
    """Load set of processed parquet file paths and skip count from checkpoint.

    Returns:
        (processed_parquets, skip_samples_in_first_parquet)
    """
    if not checkpoint_file.exists():
        return set(), 0
    try:
        with open(checkpoint_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        skip_samples = 0
        parquets = set()
        for line in lines:
            if line.startswith("skip_samples:"):
                skip_samples = int(line.split(":", 1)[1])
            else:
                parquets.add(line)
        return parquets, skip_samples
    except Exception:
        return set(), 0


def save_processed_parquet(checkpoint_file: Path, parquet_path: str):
    """Append a processed parquet path to checkpoint file."""
    with open(checkpoint_file, "a") as f:
        f.write(parquet_path + "\n")


def save_skip_samples(checkpoint_file: Path, skip_samples: int):
    """Save skip_samples to checkpoint file (update or append)."""
    if not checkpoint_file.exists():
        with open(checkpoint_file, "w") as f:
            f.write(f"skip_samples:{skip_samples}\n")
        return
    # Read existing content and update/add skip_samples line
    with open(checkpoint_file, "r") as f:
        lines = f.readlines()
    with open(checkpoint_file, "w") as f:
        found = False
        for line in lines:
            if line.strip().startswith("skip_samples:"):
                f.write(f"skip_samples:{skip_samples}\n")
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f"skip_samples:{skip_samples}\n")


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    print(f"Loading: {dataset_name} ({args.subset}/{args.split})")

    hub_endpoint = (
        os.environ.get("HF_HUB_ENDPOINT")
        or os.environ.get("HUGGINGFACE_HUB_BASE_URL")
        or os.environ.get("HF_ENDPOINT")
        or "https://huggingface.co"
    ).rstrip("/")
    file_endpoint = (os.environ.get("HF_ENDPOINT") or hub_endpoint).rstrip("/")

    def subset_to_path(subset: str) -> str:
        if subset.startswith("sample-"):
            return f"sample/{subset[len('sample-'):]}"
        return subset

    def list_repo_files(path: str):
        from urllib.parse import quote, urlparse, parse_qs
        import urllib.request

        safe_path = "/".join(quote(seg) for seg in path.split("/"))
        base = f"{hub_endpoint}/api/datasets/{dataset_name}/tree/main/{safe_path}"
        url = f"{base}?recursive=true&limit=1000"
        files = []
        while True:
            with urllib.request.urlopen(url) as response:
                data = json.load(response)
                files.extend(
                    item["path"]
                    for item in data
                    if isinstance(item, dict)
                    and item.get("type") == "file"
                    and item.get("path", "").endswith(".parquet")
                )
                link = response.headers.get("Link")
            if not link:
                break
            cursor = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    start = part.find("<") + 1
                    end = part.find(">", start)
                    next_url = part[start:end]
                    query = parse_qs(urlparse(next_url).query)
                    cursor = query.get("cursor", [None])[0]
            if not cursor:
                break
            url = f"{base}?recursive=true&limit=1000&cursor={cursor}"
        return files

    subset_path = subset_to_path(args.subset)
    repo_files = list_repo_files(subset_path)
    if not repo_files:
        raise RuntimeError(f"No parquet files found at {dataset_name}/{subset_path}")
    repo_files = sorted(repo_files)

    data_files = [
        f"{file_endpoint}/datasets/{dataset_name}/resolve/main/{path}"
        for path in repo_files
    ]

    output_path = Path(args.output_dir) / args.subset / args.split
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / "_parquet_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_path / "_processed_parquets.txt"

    # Resume support: track processed parquet files instead of sample counts
    processed_parquets = set()
    chunk_idx = 0
    skip_samples_in_first_parquet = 0  # For partial parquet resume
    if args.resume:
        # Determine starting chunk index from completed chunks
        completed = get_completed_chunks(output_path, args.tokenized_dir)
        if completed:
            chunk_idx = max(completed) + 1
            print(f"Resuming from chunk {chunk_idx:04d}")

        # Load already processed parquet files and skip count
        processed_parquets, skip_samples_in_first_parquet = load_processed_parquets(checkpoint_file)
        if processed_parquets:
            print(f"Resuming: {len(processed_parquets)} parquet files already processed")
            if skip_samples_in_first_parquet > 0:
                print(f"Will skip {skip_samples_in_first_parquet:,} samples in first unprocessed parquet")
        elif completed:
            # Checkpoint file doesn't exist but we have completed chunks
            # Need to rebuild checkpoint by estimating which parquets were processed
            print(f"No checkpoint file found, rebuilding from {len(completed)} completed chunks...")
            target_samples = (max(completed) + 1) * args.chunk_size
            cumulative = 0
            fallback_count = 0
            # Read parquet row counts to determine which ones to skip
            print(f"Scanning {len(repo_files)} parquet files to rebuild checkpoint...")
            for i, rp in enumerate(repo_files):
                local_p = cache_dir / rp
                # Try to get row count from parquet metadata without full download
                row_count = None
                parquet_url = f"{file_endpoint}/datasets/{dataset_name}/resolve/main/{rp}"
                print(f"  [{i+1}/{len(repo_files)}] {parquet_url}", flush=True)

                # Retry loop for metadata fetch
                max_metadata_retries = 3
                for attempt in range(1, max_metadata_retries + 1):
                    try:
                        print(f"    Fetching metadata (attempt {attempt})...", end=" ", flush=True)
                        # Small delay before request to avoid overwhelming the server
                        time.sleep(0.5 if attempt == 1 else 2 * attempt)
                        req = urllib.request.Request(parquet_url)
                        req.add_header('Range', 'bytes=-8')
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            footer_data = resp.read()
                            if len(footer_data) >= 8 and footer_data[-4:] == b'PAR1':
                                footer_size = struct.unpack('<I', footer_data[:4])[0]
                                # Now fetch the footer
                                req2 = urllib.request.Request(parquet_url)
                                req2.add_header('Range', f'bytes=-{footer_size + 8}')
                                with urllib.request.urlopen(req2, timeout=60) as resp2:
                                    footer_bytes = resp2.read()
                                    # Parse footer to get row count
                                    pf = pq.ParquetFile(io.BytesIO(footer_bytes), memory_map=False)
                                    row_count = pf.metadata.num_rows
                                    print(f"{row_count:,} rows", flush=True)
                                    break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < max_metadata_retries:
                            print(f"retry ({e})", flush=True)
                        else:
                            # All retries failed, use fallback
                            row_count = 700_000
                            fallback_count += 1
                            print(f"fallback to {row_count:,} (error: {e})", flush=True)

                if cumulative + row_count <= target_samples:
                    # This parquet was fully processed
                    processed_parquets.add(rp)
                    save_processed_parquet(checkpoint_file, rp)
                    cumulative += row_count
                else:
                    # This parquet was partially processed or not processed
                    skip_samples_in_first_parquet = target_samples - cumulative
                    if skip_samples_in_first_parquet > 0:
                        print(f"Will skip {skip_samples_in_first_parquet:,} samples in first unprocessed parquet")
                    break
            # Save skip_samples to checkpoint for future resumes
            save_skip_samples(checkpoint_file, skip_samples_in_first_parquet)
            print(f"Rebuilt checkpoint: {len(processed_parquets)} parquets marked as processed (cumulative: {cumulative:,} samples)")
            if fallback_count > 0:
                print(f"WARNING: {fallback_count} parquet(s) used fallback row count (700k). This may cause data duplication or loss.")
                print(f"         Consider deleting checkpoint and retrying when network is stable.")
    skipped_parquets = 0
    # Calculate initial progress for progress bar (samples already processed)
    initial_progress = chunk_idx * args.chunk_size

    # Single file mode (backward compatible)
    if args.chunk_size <= 0:
        jsonl_file = output_path / "data.jsonl"
        print(f"Saving: {jsonl_file}")
        total = SUBSET_SIZES.get(args.subset)
        count = 0
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for example in tqdm(dataset, total=total, desc="Downloading", unit=" examples"):
                f.write(json.dumps({"text": example.get("text", "")}, ensure_ascii=False) + "\n")
                count += 1
        print(f"\nSaved: {count:,} examples")
        return 0

    # Chunked mode
    chunk_buffer = []
    total_count = 0
    total = SUBSET_SIZES.get(args.subset)

    print(f"Saving chunks to: {output_path} (chunk_size={args.chunk_size:,})")

    timeout = int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120"))
    max_retries = int(os.environ.get("HF_HUB_MAX_RETRIES", "10"))

    def download_file(url: str, dest: Path, label: str, validate: bool = False) -> bool:
        """Download file with retry. Returns True if successful."""
        tmp_file = dest.with_suffix(dest.suffix + ".tmp")
        for attempt in range(1, max_retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as response, open(
                    tmp_file, "wb"
                ) as f:
                    total_bytes = response.headers.get("Content-Length")
                    total = int(total_bytes) if total_bytes and total_bytes.isdigit() else None
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=label,
                        leave=False,
                        bar_format="{desc}: {percentage:5.1f}%|{bar:20}| {n_fmt}/{total_fmt} [{rate_fmt}]",
                        mininterval=2.0,
                    ) as dpbar:
                        while True:
                            chunk = response.read(1024 * 1024 * 8)
                            if not chunk:
                                break
                            f.write(chunk)
                            dpbar.update(len(chunk))
                tmp_file.replace(dest)
                # Validate parquet file if requested
                if validate and not validate_parquet(dest):
                    print(f"\nFile validation failed: {dest.name}, retrying...")
                    dest.unlink(missing_ok=True)
                    raise ValueError("Parquet validation failed")
                return True
            except Exception as exc:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
                if dest.exists():
                    dest.unlink(missing_ok=True)
                if attempt >= max_retries:
                    print(f"\nDownload failed after {max_retries} attempts: {exc}")
                    return False
                sleep_s = min(120, 2 ** attempt)
                print(f"\nDownload failed ({attempt}/{max_retries}): {exc}. Retrying in {sleep_s}s")
                time.sleep(sleep_s)

    pbar = tqdm(
        total=total,
        initial=initial_progress,
        desc="Download",
        unit=" ex",
        bar_format="{desc}: {percentage:5.1f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        mininterval=5.0,
    )
    for url, repo_path in zip(data_files, repo_files):
        # Skip already processed parquet files
        if repo_path in processed_parquets:
            skipped_parquets += 1
            continue

        local_path = cache_dir / repo_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with validation if needed
        file_retries = 3
        for file_attempt in range(file_retries):
            if not local_path.exists():
                pbar.set_postfix_str(f"fetching {Path(repo_path).name}")
                if not download_file(url, local_path, f"parquet {Path(repo_path).name}", validate=True):
                    if file_attempt < file_retries - 1:
                        print(f"Retrying file {Path(repo_path).name}...")
                        continue
                    else:
                        print(f"Skipping corrupted file after {file_retries} attempts: {repo_path}")
                        break
            elif not validate_parquet(local_path):
                print(f"\nExisting file corrupted, re-downloading: {Path(repo_path).name}")
                local_path.unlink(missing_ok=True)
                continue

            try:
                dataset = load_dataset(
                    "parquet",
                    data_files=[str(local_path)],
                    split=args.split,
                    streaming=True,
                )
                samples_in_this_parquet = 0
                for example in dataset:
                    samples_in_this_parquet += 1
                    # Skip samples for partial parquet resume (only on first unprocessed parquet)
                    if skip_samples_in_first_parquet > 0:
                        skip_samples_in_first_parquet -= 1
                        pbar.update(1)
                        continue

                    chunk_buffer.append(
                        json.dumps({"text": example.get("text", "")}, ensure_ascii=False)
                    )
                    total_count += 1
                    pbar.update(1)

                    if len(chunk_buffer) >= args.chunk_size:
                        chunk_file = output_path / f"chunk_{chunk_idx:04d}.jsonl"
                        tmp_file = output_path / f"chunk_{chunk_idx:04d}.jsonl.tmp"
                        with open(tmp_file, "w", encoding="utf-8") as f:
                            f.write("\n".join(chunk_buffer) + "\n")
                        tmp_file.rename(chunk_file)  # Atomic rename
                        size_mb = chunk_file.stat().st_size / (1024**2)
                        pbar.set_postfix(chunk=f"{chunk_idx:04d}", size=f"{size_mb:.0f}MB")
                        chunk_buffer.clear()
                        chunk_idx += 1
                # Successfully processed: record to checkpoint and clean up
                save_processed_parquet(checkpoint_file, repo_path)
                if local_path.exists():
                    local_path.unlink(missing_ok=True)
                break
            except Exception as e:
                print(f"\nError processing {Path(repo_path).name}: {e}")
                local_path.unlink(missing_ok=True)
                if file_attempt < file_retries - 1:
                    print(f"Will retry downloading {Path(repo_path).name}...")
                else:
                    print(f"Skipping file after {file_retries} attempts: {repo_path}")


    pbar.close()

    # Write remaining
    if chunk_buffer:
        chunk_file = output_path / f"chunk_{chunk_idx:04d}.jsonl"
        tmp_file = output_path / f"chunk_{chunk_idx:04d}.jsonl.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk_buffer) + "\n")
        tmp_file.rename(chunk_file)  # Atomic rename
        print(f"Saved: {chunk_file.name} (final)")

    print(f"\nComplete: {total_count:,} new examples in {chunk_idx + 1} chunks")
    if skipped_parquets > 0:
        print(f"Skipped {skipped_parquets} already-processed parquet files")
    return 0


if __name__ == "__main__":
    exit(main())
