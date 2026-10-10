from __future__ import annotations

import unittest

from pydantic_settings import SettingsConfigDict

from app.core.config import Settings
from app.core.numerai_auth import (
    numerai_authorization,
    numerai_key_pair,
    resolve_numerai_credentials,
)
from app.core.numerai_mcp_client import NumeraiMCPClient


class IsolatedSettings(Settings):
    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="ZZ_ISOLATED_NUMERAI_",
        extra="ignore",
    )


class NumeraiAuthTests(unittest.TestCase):
    def test_bare_pair_gains_token_prefix(self):
        self.assertEqual(
            numerai_authorization("pub$sec"),
            "Token pub$sec",
        )
        self.assertEqual(numerai_key_pair("  pub$sec  "), ("pub", "sec"))

    def test_existing_token_prefix_is_stable(self):
        self.assertEqual(
            numerai_authorization("token pub$sec"),
            "Token pub$sec",
        )
        self.assertEqual(
            numerai_authorization("Bearer pub$sec"),
            "Token pub$sec",
        )

    def test_unknown_header_is_left_unchanged(self):
        self.assertEqual(numerai_authorization("Basic abc"), "Basic abc")
        self.assertIsNone(numerai_key_pair("Basic abc"))
        self.assertIsNone(numerai_authorization("  "))
        self.assertIsNone(numerai_authorization(None))

    def test_pair_fills_missing_key_fields(self):
        self.assertEqual(
            resolve_numerai_credentials("pub$sec", None, None),
            ("Token pub$sec", "pub", "sec"),
        )

    def test_explicit_keys_are_not_replaced(self):
        self.assertEqual(
            resolve_numerai_credentials("pub$sec", "other", "kept"),
            ("Token pub$sec", "other", "kept"),
        )

    def test_separate_keys_build_mcp_auth(self):
        self.assertEqual(
            resolve_numerai_credentials(None, "pub", "sec"),
            ("Token pub$sec", "pub", "sec"),
        )

    def test_settings_normalize_a_bare_mcp_auth_value(self):
        settings = IsolatedSettings(NUMERAI_MCP_AUTH="pub$sec")
        self.assertEqual(settings.NUMERAI_MCP_AUTH, "Token pub$sec")
        self.assertEqual(settings.NUMERAI_PUBLIC_ID, "pub")
        self.assertEqual(settings.NUMERAI_SECRET_KEY, "sec")

    def test_mcp_client_sends_the_token_prefix(self):
        client = NumeraiMCPClient(
            url="https://example.test/mcp/sse",
            auth_header="pub$sec",
        )
        self.assertEqual(client._headers(), {"Authorization": "Token pub$sec"})


if __name__ == "__main__":
    unittest.main()
