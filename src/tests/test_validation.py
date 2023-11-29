import unittest

from pydantic import ValidationError
from dbchat.validation import load_config


class TestConfig(unittest.TestCase):

    def test_load_valid_config(self):
        config = load_config('valid_config.yml')
        self.assertIsNotNone(config)
        self.assertEqual(config.appproach, "sql_engine_w_reranking")

    def test_load_invalid_config(self):
        with self.assertRaises(ValidationError):
            config = load_config('invalid_config.yml')


if __name__ == '__main__':
    unittest.main()
