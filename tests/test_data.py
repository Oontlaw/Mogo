import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from pypdf import PdfWriter

from mogo.data.books import read_book_text
from mogo.data.clean import clean_text
from mogo.data.dataset import TokenMemmap
from mogo.data.prepare import split_books


class DataTests(unittest.TestCase):
    def test_clean_text_preserves_paragraphs_and_removes_noise(self):
        self.assertEqual(clean_text("  A  \r\n\r\n\r\nB\x00\tC  "), "A\n\nB C")

    def test_book_split_is_reproducible_and_disjoint(self):
        books = [Path(f"book-{index}.txt") for index in range(10)]
        train, validation = split_books(books, 0.2, 7)
        self.assertEqual(validation, split_books(books, 0.2, 7)[1])
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(len(train) + len(validation), len(books))

    def test_reads_text_books(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.txt"
            path.write_text("A book", encoding="utf-8")
            self.assertEqual(read_book_text(path), "A book")

    def test_rejects_scanned_pdf_without_extractable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scanned.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with path.open("wb") as file:
                writer.write(file)
            with self.assertRaisesRegex(ValueError, "may be scanned"):
                read_book_text(path)

    def test_memmap_batches_shift_inputs_and_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            np.arange(12, dtype=np.uint16).tofile(path)
            with TokenMemmap(path) as data:
                inputs, targets = data.get_batch(
                    2, 4, torch.device("cpu"), np.random.default_rng(1)
                )
                self.assertEqual(inputs.shape, (2, 4))
                self.assertTrue(torch.equal(inputs + 1, targets))


if __name__ == "__main__":
    unittest.main()
