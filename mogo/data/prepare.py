import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from pypdf.errors import PdfReadError
from tokenizers import Tokenizer

from mogo.config import load_config
from mogo.data.books import read_book_text
from mogo.data.clean import clean_text
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
) -> tuple[list[Path], int, dict[Path, str], list[tuple[Path, str]]]:
    unique_paths = []
    text_by_path = {}
    seen_hashes = set()
    skipped_duplicates = 0
    skipped_files = []
    for path in book_paths:
        try:
            text = clean_text(read_book_text(path))
        except (OSError, ValueError, PdfReadError) as error:
            reason = " ".join(str(error).split())
            skipped_files.append((path, reason))
            continue

        content = text.encode("utf-8")
        content_hash = hashlib.sha256(content).digest()
        if content_hash in seen_hashes:
            skipped_duplicates += 1
            skipped_files.append((path, "duplicate book content"))
            continue
        seen_hashes.add(content_hash)
        unique_paths.append(path)
        text_by_path[path] = text
    return unique_paths, skipped_duplicates, text_by_path, skipped_files


def _token_count(
    paths: list[Path], tokenizer: Tokenizer, text_by_path: dict[Path, str]
) -> int:
    return sum(len(tokenizer.encode(text_by_path[path]).ids) + 1 for path in paths)


def _write_memmap(
    path: Path,
    book_paths: list[Path],
    tokenizer: Tokenizer,
    end_id: int,
    count: int,
    text_by_path: dict[Path, str],
) -> None:
    tokens = np.memmap(path, dtype=np.uint16, mode="w+", shape=(count,))
    offset = 0
    for book_path in book_paths:
        encoded = tokenizer.encode(text_by_path[book_path]).ids
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
    data_config = config["data"]
    input_dir = Path(data_config["input_dir"])
    output_dir = Path(data_config["output_dir"])
    book_paths = sorted(
        path
        for pattern in ("*.txt", "*.pdf")
        for path in input_dir.rglob(pattern)
        if path.is_file()
    )
    if not book_paths:
        raise FileNotFoundError(f"No .txt or .pdf books found under {input_dir}")
    book_paths, duplicate_count, text_by_path, skipped_files = _unique_books(book_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
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
