from collections import Counter
import re
import unicodedata


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DIGITS = re.compile(r"\d+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_PAGE_NUMBER = re.compile(r"\d{1,4}")
_GUTENBERG_START = re.compile(r"^[ \t]*\*{3}[ \t]*START OF\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_END = re.compile(r"^[ \t]*\*{3}[ \t]*END OF\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)
_KEEP_LINE_ENDINGS = (".", "!", "?", '"', "”", "’", "'")


def clean_text(text: str) -> str:
    return clean_text_with_report(text)[0]


def clean_text_with_report(text: str) -> tuple[str, int]:
    """Normalize common text-file noise while retaining paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end() :]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    text = _CONTROL_CHARS.sub("", text)
    lines = text.split("\n")
    normalized_counts = Counter(
        _DIGITS.sub("#", line.strip()) for line in lines if line.strip()
    )
    kept_lines = []
    removed_lines = 0
    for line in lines:
        stripped = line.strip()
        normalized = _DIGITS.sub("#", stripped)
        repeated_header = (
            len(normalized) >= 4
            and normalized_counts[normalized] >= 15
            and not normalized.endswith(_KEEP_LINE_ENDINGS)
        )
        if _PAGE_NUMBER.fullmatch(stripped) or repeated_header:
            removed_lines += 1
            continue
        kept_lines.append(line.rstrip())
    text = "\n".join(kept_lines)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip(), removed_lines
