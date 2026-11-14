import unittest

from mogo.train.schedule import cosine_learning_rate


class ScheduleTests(unittest.TestCase):
    def test_warmup_and_cosine_endpoints(self):
        self.assertAlmostEqual(cosine_learning_rate(0, 1.0, 0.1, 10, 100), 0.1)
        self.assertAlmostEqual(cosine_learning_rate(10, 1.0, 0.1, 10, 100), 1.0)
        self.assertAlmostEqual(cosine_learning_rate(100, 1.0, 0.1, 10, 100), 0.1)


if __name__ == "__main__":
    unittest.main()
