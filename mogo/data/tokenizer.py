from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from mogo.data.books import read_book_text
from mogo.data.clean import clean_text


END_OF_BOOK = "<|endofbook|>"


def train_tokenizer(
    book_paths: list[Path],
    vocab_size: int,
    min_frequency: int,
    text_by_path: dict[Path, str | Path] | None = None,
) -> Tokenizer:
    if not book_paths:
        raise ValueError("Cannot train a tokenizer without training books")

    def text_iterator() -> Iterator[str]:
        for path in book_paths:
            text = read_book_text(path) if text_by_path is None else text_by_path[path]
            if isinstance(text, Path):
                text = text.read_text(encoding="utf-8")
            yield clean_text(text)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[END_OF_BOOK],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    return tokenizer
