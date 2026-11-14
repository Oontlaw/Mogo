import re
import unicodedata


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_GUTENBERG_START = re.compile(r"^[ \t]*\*{3}[ \t]*START OF\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_END = re.compile(r"^[ \t]*\*{3}[ \t]*END OF\b[^\n]*(?:\n|$)", re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
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
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()
