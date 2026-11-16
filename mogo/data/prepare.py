import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from mogo.config import load_config
from mogo.data.books import BookExtraction, read_book_text_with_metrics
from mogo.data.quality import single_rate, split_rate, words
from mogo.data.tokenizer import END_OF_BOOK, train_tokenizer


def split_books(
    book_paths: list[Path], validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    if len(book_paths) < 2:
        raise ValueError("At least two .txt books are required for a book-level split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = sorted(book_paths)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def _unique_books(
    book_paths: list[Path],
    extraction_records: list[tuple[Path, BookExtraction | None, str | None]]
    | None = None,
    quality_limits: dict[str, float | int] | None = None,
    spool_dir: Path | None = None,
) -> tuple[list[Path], int, dict[Path, str | Path], list[tuple[Path, str]]]:
    unique_paths = []
    text_by_path = {}
    seen_hashes = set()
    skipped_duplicates = 0
    skipped_files = []
    for path in book_paths:
        try:
            extraction = read_book_text_with_metrics(path)
            text = extraction.text
        except Exception as error:
            reason = " ".join(str(error).split())
            if not isinstance(error, ValueError):
                reason = f"{type(error).__name__}: {reason}"
            skipped_files.append((path, reason))
            if extraction_records is not None:
                extraction_records.append((path, None, reason))
            continue

        if quality_limits is not None:
            word_count = len(words(text))
            split_value = split_rate(text)
            single_value = single_rate(text)
            skip_reason = None
            if split_value > quality_limits["max_split_rate"]:
                skip_reason = (
                    f"split_rate {split_value:.1f} > "
                    f"{quality_limits['max_split_rate']:.1f}"
                )
            elif single_value > quality_limits["max_single_rate"]:
                skip_reason = (
                    f"single_rate {single_value:.1f} > "
                    f"{quality_limits['max_single_rate']:.1f}"
                )
            elif word_count < quality_limits["min_words"]:
                skip_reason = (
                    f"word_count {word_count} < "
                    f"min_words {quality_limits['min_words']}"
                )
            if skip_reason is not None:
                skipped_files.append((path, skip_reason))
                if extraction_records is not None:
                    extraction_records.append((path, extraction, skip_reason))
                continue

        if extraction_records is not None:
            extraction_records.append((path, extraction, None))
        content = text.encode("utf-8")
        content_hash = hashlib.sha256(content).digest()
        if content_hash in seen_hashes:
            skipped_duplicates += 1
            skipped_files.append((path, "duplicate book content"))
            if extraction_records is not None:
                extraction_records[-1] = (
                    path,
                    extraction,
                    "duplicate book content",
                )
            continue
        seen_hashes.add(content_hash)
        unique_paths.append(path)
        if spool_dir is None:
            text_by_path[path] = text
        else:
            spool_path = spool_dir / f"{len(unique_paths):08d}.txt"
            spool_path.write_text(text, encoding="utf-8")
            text_by_path[path] = spool_path
    return unique_paths, skipped_duplicates, text_by_path, skipped_files


def _write_extraction_report(
    path: Path, records: list[tuple[Path, BookExtraction | None, str | None]]
) -> None:
    lines = []
    for book_path, extraction, skipped_reason in records:
        lines.append(f"{book_path}\n")
        if extraction is None:
            lines.append(f"  skipped: {skipped_reason}\n")
            continue
        lines.append(f"  chosen: {extraction.chosen_extractor}\n")
        if skipped_reason:
            lines.append(f"  skipped: {skipped_reason}\n")
        for name, metrics in extraction.candidates.items():
            lines.append(
                f"  {name}: split_rate={metrics.split_rate:.6f}, "
                f"single_rate={metrics.single_rate:.6f}, "
                f"characters={metrics.characters}, "
                f"header_lines_removed={metrics.header_lines_removed}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


def _token_count(
    paths: list[Path], tokenizer: Tokenizer, text_by_path: dict[Path, str | Path]
) -> int:
    count = 0
    for path in paths:
        text = text_by_path[path]
        if isinstance(text, Path):
            text = text.read_text(encoding="utf-8")
        count += len(tokenizer.encode(text).ids) + 1
    return count


def _write_memmap(
    path: Path,
    book_paths: list[Path],
    tokenizer: Tokenizer,
    end_id: int,
    count: int,
    text_by_path: dict[Path, str | Path],
) -> None:
    tokens = np.memmap(path, dtype=np.uint16, mode="w+", shape=(count,))
    offset = 0
    for book_path in book_paths:
        text = text_by_path[book_path]
        if isinstance(text, Path):
            text = text.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text).ids
        encoded.append(end_id)
        next_offset = offset + len(encoded)
        tokens[offset:next_offset] = encoded
        offset = next_offset
    if offset != count:
        raise RuntimeError(f"Token count changed while writing {path}: {offset} != {count}")
    tokens.flush()
    del tokens


def prepare(config_path: str | Path) -> None:
    config = load_config(config_path)
    with tempfile.TemporaryDirectory(prefix="mogo-prepare-") as spool_directory:
        _prepare(config, Path(spool_directory))


def _prepare(config: dict, spool_dir: Path) -> None:
    data_config = config["data"]
    input_dir = Path(data_config["input_dir"])
    output_dir = Path(data_config["output_dir"])
    resolved_output_dir = output_dir.resolve()
    book_paths = sorted(
        path
        for pattern in ("*.txt", "*.pdf")
        for path in input_dir.rglob(pattern)
        if path.is_file()
        and not path.resolve().is_relative_to(resolved_output_dir)
    )
    if not book_paths:
        raise FileNotFoundError(f"No .txt or .pdf books found under {input_dir}")
    quality_limits = {
        "max_split_rate": float(data_config.get("max_split_rate", 1.0)),
        "max_single_rate": float(data_config.get("max_single_rate", 10.0)),
        "min_words": int(data_config.get("min_words", 10000)),
    }
    extraction_records = []
    book_paths, duplicate_count, text_by_path, skipped_files = _unique_books(
        book_paths, extraction_records, quality_limits, spool_dir=spool_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_extraction_report(output_dir / "extraction_report.txt", extraction_records)
    skipped_path = output_dir / "skipped_books.txt"
    skipped_path.write_text(
        "".join(f"{path}\t{reason}\n" for path, reason in skipped_files),
        encoding="utf-8",
    )
    print(
        f"Skipped {len(skipped_files)} file(s), "
        f"including {duplicate_count} duplicate(s)"
    )
    for path, reason in skipped_files:
        print(f"  {path}: {reason}")
    if not book_paths:
        raise ValueError(
            f"No readable books found under {input_dir}; see {skipped_path}"
        )

    train_books, validation_books = split_books(
        book_paths,
        float(data_config["validation_fraction"]),
        int(config["seed"]),
    )
    tokenizer = train_tokenizer(
        train_books,
        int(data_config["vocab_size"]),
        int(data_config["min_token_frequency"]),
        text_by_path=text_by_path,
    )
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocabulary exceeds the uint16 token ID range")
    end_id = tokenizer.token_to_id(END_OF_BOOK)
    if end_id is None:
        raise RuntimeError(f"Tokenizer is missing its {END_OF_BOOK} special token")

    tokenizer.save(str(output_dir / "tokenizer.json"))
    counts = {}
    for name, paths in (("train", train_books), ("val", validation_books)):
        count = _token_count(paths, tokenizer, text_by_path)
        _write_memmap(
            output_dir / f"{name}.bin",
            paths,
            tokenizer,
            end_id,
            count,
            text_by_path,
        )
        counts[name] = count
        print(f"{name}: {len(paths)} books, {count:,} tokens")

    metadata = {
        "dtype": "uint16",
        "vocab_size": tokenizer.get_vocab_size(),
        "end_of_book_token_id": end_id,
        "train_books": [str(path) for path in train_books],
        "validation_books": [str(path) for path in validation_books],
        "token_counts": counts,
    }
    (output_dir / "data_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Tokenizer and memory-mapped token files written to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MOGO book data")
    parser.add_argument("--config", default="configs/mogo.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
