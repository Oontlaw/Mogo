from pathlib import Path

from pypdf import PdfReader


def read_book_text(path: str | Path) -> str:
    book_path = Path(path)
    if book_path.suffix.lower() == ".txt":
        return book_path.read_text(encoding="utf-8-sig", errors="replace")
    if book_path.suffix.lower() != ".pdf":
        raise ValueError(f"Unsupported book format: {book_path}")

    reader = PdfReader(str(book_path), strict=False)
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be read: {book_path}")

    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise ValueError(
            f"No extractable text found in {book_path}; "
            "it may be scanned and require OCR"
        )
    return text
