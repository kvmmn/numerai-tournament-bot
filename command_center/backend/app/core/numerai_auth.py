"""Normalize Numerai API credentials from either accepted env shape.

Numerai accepts ``Authorization: Token PUBLIC_ID$SECRET_KEY``. Operators
sometimes store only ``PUBLIC_ID$SECRET_KEY`` in ``NUMERAI_MCP_AUTH``.
"""

from __future__ import annotations

import re

_PAIR = re.compile(r"^[A-Za-z0-9_-]+\$[A-Za-z0-9_-]+$")


def numerai_key_pair(value: str | None) -> tuple[str, str] | None:
    """Return ``(public_id, secret_key)`` when ``value`` is a Numerai key pair."""
    if value is None:
        return None
    token = value.strip()
    lowered = token.lower()
    if lowered.startswith("token "):
        token = token.split(None, 1)[1].strip()
    elif lowered.startswith("bearer "):
        token = token.split(None, 1)[1].strip()
    if not _PAIR.fullmatch(token):
        return None
    public_id, secret_key = token.split("$", 1)
    return public_id, secret_key


def numerai_authorization(value: str | None) -> str | None:
    """Return an Authorization header value, adding the Token prefix when needed.

    Unrecognized values are returned stripped so a future header format is not
    rewritten into something Numerai would reject.
    """
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    pair = numerai_key_pair(stripped)
    if pair is None:
        return stripped
    public_id, secret_key = pair
    return f"Token {public_id}${secret_key}"


def resolve_numerai_credentials(
    mcp_auth: str | None,
    public_id: str | None,
    secret_key: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Fill whichever credential form is missing from the other."""
    header = numerai_authorization(mcp_auth)
    pair = numerai_key_pair(header)
    if pair is not None:
        parsed_public, parsed_secret = pair
        public_id = public_id or parsed_public
        secret_key = secret_key or parsed_secret
    elif public_id and secret_key and not header:
        header = numerai_authorization(f"{public_id}${secret_key}")
    return header, public_id, secret_key
