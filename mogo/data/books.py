from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from mogo.data.clean import clean_text_with_report
from mogo.data.quality import single_rate, split_rate


@dataclass(frozen=True)
class ExtractionMetrics:
    characters: int
    single_rate: float
    split_rate: float
    header_lines_removed: int = 0


@dataclass(frozen=True)
class BookExtraction:
    text: str
    chosen_extractor: str
    candidates: dict[str, ExtractionMetrics]


def _load_pymupdf():
    try:
        import pymupdf
    except ImportError:
        return None
    return pymupdf


def _measure_candidate(text: str) -> tuple[str, ExtractionMetrics]:
    cleaned, removed_lines = clean_text_with_report(text)
    metrics = ExtractionMetrics(
        characters=len(cleaned),
        single_rate=single_rate(cleaned),
        split_rate=split_rate(cleaned),
        header_lines_removed=removed_lines,
    )
    return cleaned, metrics


def read_book_text_with_metrics(path: str | Path) -> BookExtraction:
    book_path = Path(path)
    if book_path.suffix.lower() == ".txt":
        raw_text = book_path.read_text(encoding="utf-8-sig", errors="replace")
        text, metrics = _measure_candidate(raw_text)
        return BookExtraction(text, "text", {"text": metrics})
    if book_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported book format: {book_path}")

    reader = PdfReader(str(book_path), strict=False)
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be read: {book_path}")

    pypdf_text = "\n\n".join(
        page_text
        for page_text in (page.extract_text() or "" for page in reader.pages)
        if page_text.strip()
    )
    cleaned_candidates = {}
    candidates = {}
    cleaned_candidates["pypdf"], candidates["pypdf"] = _measure_candidate(pypdf_text)

    pymupdf = _load_pymupdf()
    if pymupdf is not None:
        with pymupdf.open(str(book_path)) as document:
            pymupdf_text = "\n\n".join(
                page.get_text("text") or "" for page in document
            )
        cleaned_candidates["pymupdf"], candidates["pymupdf"] = _measure_candidate(
            pymupdf_text
        )

    available = [
        name for name, text in cleaned_candidates.items() if text.strip()
    ]
    if not available:
        raise ValueError(
            f"No extractable text found in {book_path}; "
            "it may be scanned and require OCR"
        )

    chosen = min(
        available,
        key=lambda name: (
            candidates[name].split_rate,
            candidates[name].single_rate,
            -candidates[name].characters,
        ),
    )
    return BookExtraction(cleaned_candidates[chosen], chosen, candidates)


def read_book_text(path: str | Path) -> str:
    return read_book_text_with_metrics(path).text
