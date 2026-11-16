import argparse
import csv
import json
import math
import re
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from mogo.config import load_config


_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_ENGLISH_WORDS = frozenset(
    "a about after all also an and any are as at be because been before being "
    "between both but by can come could day did do down each even find first for "
    "from get give go good had has have he her here him his how i if in into is "
    "it its just know like little long look made make many may me more most much "
    "must my new no not now of off on one only or other our out over people said "
    "same see she should so some take than that the their them then there these "
    "they think this time to two up use very was way we well were what when which "
    "who will with would you your".split()
)
_USER_AGENT = "MOGO-corpus-builder/0.1 (polite dataset download)"
_TINY_STORY_END = {"<|endoftext|>", "[END]"}


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _is_english(text: str) -> bool:
    tokens = _WORD_RE.findall(text)
    if len(tokens) < 20:
        return False
    all_letters = sum(character.isalpha() for character in text)
    ascii_letters = sum(character.isascii() and character.isalpha() for character in text)
    common_words = sum(token.lower() in _ENGLISH_WORDS for token in tokens)
    return common_words / len(tokens) >= 0.1 and ascii_letters / max(1, all_letters) >= 0.95


def _open_url(url: str):
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    return urlopen(request, timeout=60)


def _download_bytes(url: str, max_bytes: int) -> tuple[bytes, int]:
    result = bytearray()
    with _open_url(url) as response:
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            if len(result) + len(chunk) > max_bytes:
                raise RuntimeError(f"Download byte limit exceeded for {url}")
            result.extend(chunk)
    return bytes(result), len(result)


def _read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a list in corpus manifest {path}")
    return value


def _write_manifest(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            json.dump(records, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(text)
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _record(
    source: str,
    original_id: str,
    license_name: str,
    file_path: Path,
    output_dir: Path,
    text: str,
    tokens_per_word: float,
    download_bytes: int,
    url: str | None = None,
) -> dict:
    record = {
        "source": source,
        "original_id": original_id,
        "license": license_name,
        "word_count": _word_count(text),
        "estimated_tokens": math.ceil(_word_count(text) * tokens_per_word),
        "file_path": str(file_path.relative_to(output_dir.parent)),
        "download_bytes": download_bytes,
    }
    if url:
        record["url"] = url
    return record


def _source_totals(records: list[dict], source: str) -> tuple[int, int]:
    matching = [record for record in records if record["source"] == source]
    return (
        sum(int(record["estimated_tokens"]) for record in matching),
        sum(int(record.get("download_bytes", 0)) for record in matching),
    )


def _remaining_tokens(
    records: list[dict], source: str, source_cap: int, total_cap: int
) -> int:
    source_tokens, _ = _source_totals(records, source)
    all_tokens = sum(int(record["estimated_tokens"]) for record in records)
    return max(0, min(source_cap - source_tokens, total_cap - all_tokens))


def _download_budget(
    records: list[dict], source: str, source_byte_cap: int, total_byte_cap: int
) -> int:
    _, source_bytes = _source_totals(records, source)
    total_bytes = sum(int(record.get("download_bytes", 0)) for record in records)
    return max(0, min(source_byte_cap - source_bytes, total_byte_cap - total_bytes))


def _write_record(
    records: list[dict], manifest_path: Path, record: dict, file_path: Path
) -> None:
    _write_text_atomic(file_path, record.pop("_text"))
    records.append(record)
    _write_manifest(manifest_path, records)


def _pg19_items(
    source_config: dict,
    source_name: str,
    output_dir: Path,
    manifest_path: Path,
    records: list[dict],
    total_token_cap: int,
    total_byte_cap: int,
    tokens_per_word: float,
) -> None:
    source_cap = int(source_config["token_cap"])
    source_byte_cap = int(source_config["max_download_bytes"])
    if _remaining_tokens(records, source_name, source_cap, total_token_cap) <= 0:
        return
    list_url = source_config["train_files_url"]
    names_bytes, _ = _download_bytes(list_url, min(1_000_000, total_byte_cap))
    filenames = sorted(
        line.strip()
        for line in names_bytes.decode("utf-8").splitlines()
        if line.strip()
    )
    metadata_url = source_config["metadata_url"]
    budget = _download_budget(records, source_name, source_byte_cap, total_byte_cap)
    metadata_bytes, metadata_size = _download_bytes(metadata_url, budget)
    metadata = {
        row["_id"]: row
        for row in csv.DictReader(
            metadata_bytes.decode("utf-8").splitlines(),
            fieldnames=["_id", "short_book_title", "publication_date", "url"],
        )
    }
    remaining = _remaining_tokens(records, source_name, source_cap, total_token_cap)
    downloaded = metadata_size
    saved = 0
    for filename in filenames:
        if remaining <= 0:
            break
        original_id = Path(filename).stem
        file_path = output_dir / source_name / f"{original_id}.txt"
        if file_path.exists():
            if not any(record.get("file_path") == str(file_path.relative_to(output_dir.parent)) for record in records):
                existing_text = file_path.read_text(encoding="utf-8")
                records.append(
                    _record(
                        source_name,
                        original_id,
                        source_config["license"],
                        file_path,
                        output_dir,
                        existing_text,
                        tokens_per_word,
                        0,
                        metadata.get(original_id, {}).get("url"),
                    )
                )
                _write_manifest(manifest_path, records)
            continue
        byte_budget = min(
            _download_budget(records, source_name, source_byte_cap, total_byte_cap),
            max(0, source_byte_cap - downloaded),
        )
        if byte_budget <= 0:
            print(f"{source_name}: download byte cap reached")
            break
        url = f"{source_config['book_base_url']}{filename}"
        try:
            raw_bytes, size = _download_bytes(url, byte_budget)
        except RuntimeError as error:
            if "byte limit exceeded" in str(error):
                print(f"{source_name}: download byte cap reached")
                break
            raise
        downloaded += size
        text = raw_bytes.decode("utf-8", errors="replace")
        word_count = _word_count(text)
        if word_count < int(source_config.get("min_words", 10000)):
            print(f"{source_name}: skipped {original_id}, only {word_count:,} words")
            continue
        if not _is_english(text):
            print(f"{source_name}: skipped {original_id}, English filter")
            continue
        if math.ceil(word_count * tokens_per_word) > remaining:
            print(f"{source_name}: token cap reached before {original_id}")
            break
        details = metadata.get(original_id, {})
        cleaned_text = text
        record = _record(
            source_name,
            original_id,
            source_config["license"],
            file_path,
            output_dir,
            cleaned_text,
            tokens_per_word,
            size,
            details.get("url"),
        )
        record["_text"] = cleaned_text
        _write_record(records, manifest_path, record, file_path)
        remaining -= record["estimated_tokens"]
        saved += 1
        if saved % 10 == 0:
            source_tokens, _ = _source_totals(records, source_name)
            print(
                f"{source_name}: saved {saved} books, "
                f"{source_tokens:,} estimated tokens, {downloaded:,} bytes fetched"
            )
    source_tokens, source_bytes = _source_totals(records, source_name)
    print(
        f"{source_name}: complete with {source_tokens:,} estimated tokens "
        f"({source_bytes:,} manifest bytes)"
    )


def _iter_tiny_stories(url: str, byte_budget: int):
    buffer = []
    byte_count = 0
    with _open_url(url) as response:
        while True:
            line = response.readline()
            if not line:
                break
            byte_count += len(line)
            if byte_count > byte_budget:
                raise RuntimeError(f"Download byte limit exceeded for {url}")
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text in _TINY_STORY_END:
                story = "\n".join(buffer).strip()
                if story:
                    yield story, byte_count
                buffer.clear()
            else:
                buffer.append(text)
    story = "\n".join(buffer).strip()
    if story:
        yield story, byte_count


def _tiny_stories(
    source_config: dict,
    source_name: str,
    output_dir: Path,
    manifest_path: Path,
    records: list[dict],
    total_token_cap: int,
    total_byte_cap: int,
    tokens_per_word: float,
) -> None:
    source_cap = int(source_config["token_cap"])
    source_byte_cap = int(source_config["max_download_bytes"])
    remaining = _remaining_tokens(records, source_name, source_cap, total_token_cap)
    if remaining <= 0:
        return
    existing = {
        int(Path(record["file_path"]).stem.removeprefix("batch-"))
        for record in records
        if record["source"] == source_name
    }
    next_batch = max(existing, default=0) + 1
    batch_words = int(source_config.get("batch_words", 100000))
    stories = []
    words_in_batch = 0
    word_offset = sum(
        int(record["word_count"])
        for record in records
        if record["source"] == source_name
    )
    words_to_skip = word_offset
    downloaded_total = 0
    last_written_bytes = 0

    def save_batch() -> None:
        nonlocal stories, words_in_batch, next_batch, remaining, word_offset
        nonlocal last_written_bytes
        if not stories or remaining <= 0:
            return
        text = "\n\n".join(stories).strip()
        path = output_dir / source_name / f"batch-{next_batch:05d}.txt"
        if path.exists():
            stories = []
            words_in_batch = 0
            next_batch += 1
            return
        batch_word_count = _word_count(text)
        record = _record(
            source_name,
            f"stories-{word_offset + 1}-{word_offset + batch_word_count}",
            source_config["license"],
            path,
            output_dir,
            text,
            tokens_per_word,
            max(0, downloaded_total - last_written_bytes),
            source_config.get("url"),
        )
        record["_text"] = text
        _write_record(records, manifest_path, record, path)
        last_written_bytes = downloaded_total
        remaining -= record["estimated_tokens"]
        word_offset += batch_word_count
        next_batch += 1
        stories = []
        words_in_batch = 0
        if next_batch % 10 == 0:
            source_tokens, _ = _source_totals(records, source_name)
            print(f"{source_name}: saved {next_batch - 1} batches, {source_tokens:,} estimated tokens")

    budget = _download_budget(records, source_name, source_byte_cap, total_byte_cap)
    for story, bytes_read in _iter_tiny_stories(source_config["url"], budget):
        downloaded_total = max(downloaded_total, bytes_read)
        if not _is_english(story):
            continue
        story_words = _word_count(story)
        if words_to_skip:
            if story_words <= words_to_skip:
                words_to_skip -= story_words
                continue
            matches = list(_WORD_RE.finditer(story))
            story = story[matches[words_to_skip - 1].end() :].lstrip()
            story_words = _word_count(story)
            words_to_skip = 0
        remaining_words = max(
            0, int(remaining / tokens_per_word) - words_in_batch
        )
        if remaining_words <= 0:
            save_batch()
            break
        if story_words > remaining_words:
            matches = list(_WORD_RE.finditer(story))
            if matches:
                story = story[: matches[remaining_words - 1].end()]
                story_words = _word_count(story)
        stories.append(story)
        words_in_batch += story_words
        if words_in_batch >= batch_words or math.ceil(words_in_batch * tokens_per_word) >= remaining:
            save_batch()
        if remaining <= 0:
            break
    save_batch()
    source_tokens, source_bytes = _source_totals(records, source_name)
    print(
        f"{source_name}: complete with {source_tokens:,} estimated tokens "
        f"({source_bytes:,} manifest bytes; {downloaded_total:,} bytes streamed)"
    )


def build_corpus(config_path: str | Path) -> None:
    started = time.monotonic()
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    manifest_path = output_dir / "manifest.json"
    records = _read_manifest(manifest_path)
    total_cap = int(config.get("total_token_cap", 300_000_000))
    total_byte_cap = int(config.get("max_download_bytes", 10_000_000_000))
    tokens_per_word = float(config.get("tokens_per_word", 1.3))
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, source_config in config["sources"].items():
        if not source_config.get("enabled", True):
            continue
        if sum(int(record["estimated_tokens"]) for record in records) >= total_cap:
            print(f"Total token cap ({total_cap:,}) reached")
            break
        print(f"Building {source_name}")
        if source_config["type"] == "pg19":
            _pg19_items(
                source_config,
                source_name,
                output_dir,
                manifest_path,
                records,
                total_cap,
                total_byte_cap,
                tokens_per_word,
            )
        elif source_config["type"] == "tiny_stories":
            _tiny_stories(
                source_config,
                source_name,
                output_dir,
                manifest_path,
                records,
                total_cap,
                total_byte_cap,
                tokens_per_word,
            )
        else:
            raise ValueError(f"Unsupported corpus source type: {source_config['type']}")
    _write_manifest(manifest_path, records)
    elapsed = time.monotonic() - started
    print(f"Corpus build complete: {len(records)} files in {elapsed:.1f}s")
    for source_name in config["sources"]:
        items = [record for record in records if record["source"] == source_name]
        print(
            f"{source_name}: {len(items)} files, "
            f"{sum(int(record['word_count']) for record in items):,} words, "
            f"{sum(int(record['estimated_tokens']) for record in items):,} estimated tokens"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MOGO training corpus")
    parser.add_argument("--config", default="configs/corpus.yaml")
    args = parser.parse_args()
    build_corpus(args.config)


if __name__ == "__main__":
    main()
