import re
from collections import Counter


_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_EXCLUDED_SINGLE_LETTERS = {"a", "A", "I"}


def words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text)


def single_rate(text: str) -> float:
    tokens = words(text)
    if not tokens:
        return 0.0
    single_letters = sum(
        len(token) == 1 and token not in _EXCLUDED_SINGLE_LETTERS
        for token in tokens
    )
    return single_letters * 1000 / len(tokens)


def split_rate(text: str) -> float:
    tokens = words(text)
    if not tokens:
        return 0.0
    frequencies = Counter(tokens)
    split_words = sum(
        len(left) == 1
        and left not in _EXCLUDED_SINGLE_LETTERS
        and right.islower()
        and frequencies[left + right] >= 5
        for left, right in zip(tokens, tokens[1:])
    )
    return split_words * 1000 / len(tokens)
