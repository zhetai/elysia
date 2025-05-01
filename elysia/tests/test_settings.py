import os
import unittest
from elysia.config import Settings, configure

from elysia.config import settings as global_settings
from elysia.config import reset_settings


class TestConfig(unittest.TestCase):

    def reset_global_settings(self):
        reset_settings()

    def test_setup(self):
        """
        Test that settings works
        """
        settings = Settings()

        # LLM should not be set by default
        self.assertIsNone(settings.BASE_MODEL)
        self.assertIsNone(settings.COMPLEX_MODEL)
        self.assertIsNone(settings.BASE_PROVIDER)
        self.assertIsNone(settings.COMPLEX_PROVIDER)
        self.assertIsNone(settings.MODEL_API_BASE)

        self.reset_global_settings()

    def test_default_config(self):
        """
        Test that the default config is correct
        """
        settings = Settings.from_default()
        self.assertIsNotNone(settings.BASE_MODEL)
        self.assertIsNotNone(settings.COMPLEX_MODEL)
        self.assertIsNotNone(settings.BASE_PROVIDER)
        self.assertIsNotNone(settings.COMPLEX_PROVIDER)

        self.reset_global_settings()

    def test_from_env_vars(self):
        """
        Test that the config can be loaded from env vars
        """
        os.environ["BASE_MODEL"] = os.getenv("BASE_MODEL", "gemini-2.0-flash-001")
        os.environ["COMPLEX_MODEL"] = os.getenv("COMPLEX_MODEL", "gemini-2.0-flash-001")
        os.environ["BASE_PROVIDER"] = os.getenv("BASE_PROVIDER", "google")
        os.environ["COMPLEX_PROVIDER"] = os.getenv("COMPLEX_PROVIDER", "google")
        os.environ["MODEL_API_BASE"] = os.getenv(
            "MODEL_API_BASE", "https://api.google.com"
        )
        os.environ["LOCAL"] = os.getenv("LOCAL", "1")
        os.environ["WCD_URL"] = os.getenv("WCD_URL", "http://test_url.com")
        os.environ["WCD_API_KEY"] = os.getenv("WCD_API_KEY", "test_key")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "test_key_openai")
        os.environ["ANTHROPIC_API_KEY"] = os.getenv(
            "ANTHROPIC_API_KEY", "test_key_anthropic"
        )

        settings = Settings.from_env_vars()
        self.assertEqual(settings.BASE_MODEL, os.environ["BASE_MODEL"])
        self.assertEqual(settings.COMPLEX_MODEL, os.environ["COMPLEX_MODEL"])
        self.assertEqual(settings.BASE_PROVIDER, os.environ["BASE_PROVIDER"])
        self.assertEqual(settings.COMPLEX_PROVIDER, os.environ["COMPLEX_PROVIDER"])
        self.assertEqual(settings.MODEL_API_BASE, os.environ["MODEL_API_BASE"])
        self.assertEqual(settings.LOCAL, os.environ["LOCAL"] == "1")
        self.assertEqual(settings.WCD_URL, os.environ["WCD_URL"])
        self.assertEqual(settings.WCD_API_KEY, os.environ["WCD_API_KEY"])
        self.assertEqual(
            settings.API_KEYS["openai_api_key"], os.environ["OPENAI_API_KEY"]
        )
        self.assertEqual(
            settings.API_KEYS["anthropic_api_key"], os.environ["ANTHROPIC_API_KEY"]
        )

        self.reset_global_settings()

    def test_basic_configure_global(self):
        """
        Does using elysia.configure() change the global settings (for backend)
        """

        configure(base_model="gpt-4o-mini", base_provider="openai")
        self.assertEqual(global_settings.BASE_MODEL, "gpt-4o-mini")

        configure(base_model="gpt-4o", base_provider="openai")
        self.assertNotEqual(global_settings.BASE_MODEL, "gpt-4o-mini")
        self.assertEqual(global_settings.BASE_MODEL, "gpt-4o")

        configure(complex_model="claude-3-5-haiku-latest", complex_provider="anthropic")
        self.assertEqual(global_settings.COMPLEX_MODEL, "claude-3-5-haiku-latest")
        self.assertEqual(global_settings.COMPLEX_PROVIDER, "anthropic")

        self.reset_global_settings()

    def test_basic_configure_local(self):
        """
        Does using elysia.configure() change the local settings (for backend)
        """
        settings = Settings()

        settings.configure(
            complex_model="claude-3-5-haiku-latest", complex_provider="anthropic"
        )
        self.assertEqual(settings.COMPLEX_MODEL, "claude-3-5-haiku-latest")
        self.assertEqual(settings.COMPLEX_PROVIDER, "anthropic")
        self.assertNotEqual(settings.COMPLEX_MODEL, global_settings.COMPLEX_MODEL)
        self.assertNotEqual(settings.COMPLEX_PROVIDER, global_settings.COMPLEX_PROVIDER)

        settings.configure(base_model="gpt-4o-mini", base_provider="openai")
        self.assertEqual(settings.BASE_MODEL, "gpt-4o-mini")
        self.assertEqual(settings.BASE_PROVIDER, "openai")
        self.assertNotEqual(settings.BASE_MODEL, global_settings.BASE_MODEL)
        self.assertNotEqual(settings.BASE_PROVIDER, global_settings.BASE_PROVIDER)

        settings.configure(base_model="gpt-4o", base_provider="openai")
        self.assertNotEqual(settings.BASE_MODEL, "gpt-4o-mini")
        self.assertEqual(settings.BASE_MODEL, "gpt-4o")
        self.assertEqual(settings.BASE_PROVIDER, "openai")
        self.assertNotEqual(settings.BASE_MODEL, global_settings.BASE_MODEL)
        self.assertNotEqual(settings.BASE_PROVIDER, global_settings.BASE_PROVIDER)

        self.reset_global_settings()


if __name__ == "__main__":
    unittest.main()
    # TestConfig().test_setup()
