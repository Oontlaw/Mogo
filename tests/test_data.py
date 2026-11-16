import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import yaml
from pypdf import PdfWriter

from mogo.data.books import read_book_text
from mogo.data.clean import clean_text
from mogo.data.dataset import TokenMemmap
from mogo.data.prepare import _unique_books, prepare, split_books


class DataTests(unittest.TestCase):
    def test_clean_text_preserves_paragraphs_and_removes_noise(self):
        self.assertEqual(clean_text("  A  \r\n\r\n\r\nB\x00\tC  "), "A\n\nB C")

    def test_clean_text_strips_gutenberg_header_and_footer(self):
        text = (
            "Preamble\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
            "Title\n\nBody text.\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
            "Footer\n"
        )
        self.assertEqual(clean_text(text), "Title\n\nBody text.")

    def test_clean_text_handles_this_start_and_the_end_variants(self):
        text = (
            "*** START OF THIS PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
            "Body text.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
        )
        self.assertEqual(clean_text(text), "Body text.")

    def test_duplicate_books_are_skipped_by_cleaned_content(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.txt"
            duplicate = Path(directory) / "duplicate.txt"
            different = Path(directory) / "different.txt"
            first.write_text("Same book\n", encoding="utf-8")
            duplicate.write_text("Same book\n", encoding="utf-8")
            different.write_text("Another book\n", encoding="utf-8")
            unique, skipped, _, skipped_files = _unique_books(
                [first, duplicate, different]
            )
            self.assertEqual(unique, [first, different])
            self.assertEqual(skipped, 1)
            self.assertEqual(skipped_files, [(duplicate, "duplicate book content")])

    def test_prepare_skips_empty_pdf_and_unreadable_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "books"
            output_dir = root / "processed"
            input_dir.mkdir()
            for index in range(2):
                (input_dir / f"book-{index}.txt").write_text(
                    (f"Distinct book {index} text for tokenizer training. " * 20),
                    encoding="utf-8",
                )
            empty_pdf = input_dir / "empty.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)
            with empty_pdf.open("wb") as file:
                writer.write(file)
            unreadable = input_dir / "unreadable.txt"
            unreadable.write_text("unreadable", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 17,
                        "data": {
                            "input_dir": str(input_dir),
                            "output_dir": str(output_dir),
                            "validation_fraction": 0.5,
                            "vocab_size": 300,
                            "min_token_frequency": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            from mogo.data import prepare as prepare_module

            read_book_text = prepare_module.read_book_text

            def read_book(path):
                if Path(path) == unreadable:
                    raise PermissionError("test unreadable file")
                return read_book_text(path)

            with patch("mogo.data.prepare.read_book_text", side_effect=read_book):
                prepare(config_path)

            skipped = (output_dir / "skipped_books.txt").read_text(encoding="utf-8")
            self.assertIn(f"{empty_pdf}\t", skipped)
            self.assertIn("may be scanned", skipped)
            self.assertIn(f"{unreadable}\ttest unreadable file", skipped)
            self.assertTrue((output_dir / "train.bin").is_file())
            self.assertTrue((output_dir / "val.bin").is_file())

    def test_prepare_errors_when_every_book_is_unreadable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "books"
            output_dir = root / "processed"
            input_dir.mkdir()
            (input_dir / "unreadable.txt").write_text("bad", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 17,
                        "data": {
                            "input_dir": str(input_dir),
                            "output_dir": str(output_dir),
                            "validation_fraction": 0.5,
                            "vocab_size": 300,
                            "min_token_frequency": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "mogo.data.prepare.read_book_text",
                side_effect=PermissionError("test unreadable file"),
            ):
                with self.assertRaisesRegex(ValueError, "No readable books"):
                    prepare(config_path)
            self.assertIn(
                "test unreadable file",
                (output_dir / "skipped_books.txt").read_text(encoding="utf-8"),
            )

    def test_prepare_creates_outputs_from_three_text_books(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "books"
            output_dir = root / "processed"
            input_dir.mkdir()
            for index in range(3):
                (input_dir / f"book-{index}.txt").write_text(
                    (f"Book {index} has a different story. " * 20),
                    encoding="utf-8",
                )
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "seed": 17,
                        "data": {
                            "input_dir": str(input_dir),
                            "output_dir": str(output_dir),
                            "validation_fraction": 0.34,
                            "vocab_size": 300,
                            "min_token_frequency": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            prepare(config_path)
            for name in ("tokenizer.json", "train.bin", "val.bin", "data_meta.json"):
                self.assertTrue((output_dir / name).is_file(), name)
            metadata = json.loads(
                (output_dir / "data_meta.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(metadata["train_books"]), 2)
            self.assertEqual(len(metadata["validation_books"]), 1)
            self.assertEqual(metadata["dtype"], "uint16")

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
