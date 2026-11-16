import unittest

from mogo.data.quality import single_rate, split_rate, words


class QualityTests(unittest.TestCase):
    def test_words_keeps_contractions_together(self):
        self.assertEqual(
            words("can't don’t A1 e-mail"),
            ["can't", "don’t", "A", "e", "mail"],
        )

    def test_single_rate_excludes_a_A_and_I(self):
        text = "a A I b word"
        self.assertEqual(single_rate(text), 200.0)

    def test_split_rate_requires_lowercase_word_and_five_standalone_matches(self):
        text = "t he the the the the the a A I can't"
        self.assertAlmostEqual(split_rate(text), 1000 / 11)
        self.assertEqual(split_rate("t He The The The The The"), 0.0)
        self.assertEqual(split_rate("t he the the the the"), 0.0)

    def test_rates_are_zero_for_text_without_words(self):
        self.assertEqual(single_rate("123 !!!"), 0.0)
        self.assertEqual(split_rate("123 !!!"), 0.0)


if __name__ == "__main__":
    unittest.main()
