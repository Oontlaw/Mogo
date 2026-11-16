import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import yaml

from mogo.corpus.build import _is_english, build_corpus


def _story(repeats=30):
    return " ".join(
        "The little boy and his mother went to the house and looked at the garden."
        for _ in range(repeats)
    )


class CorpusBuilderTests(unittest.TestCase):
    def _config(self, root, source, **overrides):
        output_dir = root / "data" / "corpus"
        config = {
            "output_dir": str(output_dir),
            "total_token_cap": 100000,
            "tokens_per_word": 1.3,
            "max_download_bytes": 1000000,
            "sources": {source["name"]: source["config"]},
        }
        config.update(overrides)
        config_path = root / "corpus.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path, output_dir

    def test_english_filter_accepts_english_and_rejects_non_english(self):
        self.assertTrue(_is_english(_story()))
        spanish = " ".join(
            "El niño y su madre fueron a la casa y miraron el jardín."
            for _ in range(30)
        )
        self.assertFalse(_is_english(spanish))

    def test_tiny_stories_resume_skips_completed_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "name": "stories",
                "config": {
                    "enabled": True,
                    "type": "tiny_stories",
                    "token_cap": 10000,
                    "max_download_bytes": 100000,
                    "batch_words": 200,
                    "license": "test-license",
                    "url": "https://example.test/stories.txt",
                },
            }
            config_path, output_dir = self._config(root, source)
            body = (_story(3) + "\n\n" + _story(3) + "\n\n").encode("utf-8")
            with patch(
                "mogo.corpus.build._open_url", return_value=BytesIO(body)
            ):
                build_corpus(config_path)
            first_files = sorted((output_dir / "stories").glob("batch-*.txt"))
            first_content = [path.read_text(encoding="utf-8") for path in first_files]
            with patch(
                "mogo.corpus.build._open_url", return_value=BytesIO(body)
            ):
                build_corpus(config_path)
            second_files = sorted((output_dir / "stories").glob("batch-*.txt"))
            self.assertEqual(second_files, first_files)
            self.assertEqual(
                [path.read_text(encoding="utf-8") for path in second_files],
                first_content,
            )
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest), len(first_files))

    def test_pg19_skips_books_below_minimum_word_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "name": "pg19",
                "config": {
                    "enabled": True,
                    "type": "pg19",
                    "token_cap": 10000,
                    "max_download_bytes": 100000,
                    "min_words": 10000,
                    "license": "test-license",
                    "train_files_url": "https://example.test/train_files.txt",
                    "metadata_url": "https://example.test/metadata.csv",
                    "book_base_url": "https://example.test/",
                },
            }
            config_path, output_dir = self._config(root, source)
            responses = {
                source["config"]["train_files_url"]: b"train/1.txt\n",
                source["config"]["metadata_url"]: b"1,Test Book,1900,https://example.test/book\n",
                "https://example.test/train/1.txt": (_story(2)).encode("utf-8"),
            }

            def download(url, _max_bytes):
                value = responses[url]
                return value, len(value)

            with patch("mogo.corpus.build._download_bytes", side_effect=download):
                build_corpus(config_path)
            self.assertFalse((output_dir / "pg19" / "1.txt").exists())
            self.assertEqual(
                json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")),
                [],
            )

    def test_tiny_stories_respects_token_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "name": "stories",
                "config": {
                    "enabled": True,
                    "type": "tiny_stories",
                    "token_cap": 100,
                    "max_download_bytes": 100000,
                    "batch_words": 1000,
                    "license": "test-license",
                    "url": "https://example.test/stories.txt",
                },
            }
            config_path, output_dir = self._config(
                root, source, total_token_cap=100
            )
            body = (_story(100) + "\n\n").encode("utf-8")
            with patch(
                "mogo.corpus.build._open_url", return_value=BytesIO(body)
            ):
                build_corpus(config_path)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertLessEqual(
                sum(item["estimated_tokens"] for item in manifest), 100
            )

    def test_tiny_stories_cap_accounts_for_unflushed_batch_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = {
                "name": "stories",
                "config": {
                    "enabled": True,
                    "type": "tiny_stories",
                    "token_cap": 200,
                    "max_download_bytes": 100000,
                    "batch_words": 1000,
                    "license": "test-license",
                    "url": "https://example.test/stories.txt",
                },
            }
            config_path, output_dir = self._config(
                root, source, total_token_cap=200
            )
            body = (_story(6) + "\n\n" + _story(6) + "\n\n").encode("utf-8")
            with patch(
                "mogo.corpus.build._open_url", return_value=BytesIO(body)
            ):
                build_corpus(config_path)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertLessEqual(
                sum(item["estimated_tokens"] for item in manifest), 200
            )


if __name__ == "__main__":
    unittest.main()
