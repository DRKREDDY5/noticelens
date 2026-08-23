"""Secret-safe Phase 3 configuration loading.

Environment variables take precedence.  For local development, the only
dotenv file loaded by default is ``~/.noticelens.env``, outside the OneDrive
project.  This module never auto-discovers a project ``.env`` file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_SECRETS_PATH = Path.home() / ".noticelens.env"
REQUIRED_SECRET_NAMES = ("NEBIUS_API_KEY", "PINECONE_API_KEY")


class ConfigurationError(RuntimeError):
    """Raised when required configuration is absent or stored unsafely."""


@dataclass(frozen=True)
class Phase3Config:
    """Runtime configuration whose representation intentionally hides keys."""

    nebius_api_key: str = field(repr=False)
    pinecone_api_key: str = field(repr=False)
    secrets_path: Path | None = None
    nebius_base_url: str = "https://api.tokenfactory.nebius.com/v1"

    def public_summary(self) -> dict[str, str | None]:
        """Return only non-secret configuration suitable for reports."""

        return {
            "nebius_base_url": self.nebius_base_url,
            "secrets_source": "environment" if self.secrets_path is None else "external_local_file",
        }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_phase3_config(
    *,
    secret_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    project_root: Path | None = None,
) -> Phase3Config:
    """Load the two provider keys without exposing their values.

    ``environ`` is injectable for offline tests. Missing values may be read
    from one explicit external dotenv file. Project-local secret files are
    rejected when ``project_root`` is supplied.
    """

    source = dict(os.environ if environ is None else environ)
    selected_path: Path | None = None
    missing = [name for name in REQUIRED_SECRET_NAMES if not source.get(name, "").strip()]

    if missing:
        candidate = (secret_path or DEFAULT_SECRETS_PATH).expanduser().resolve()
        effective_project_root = (
            project_root.resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        if _is_within(candidate, effective_project_root):
            raise ConfigurationError("Refusing to load API secrets from inside the project workspace")
        if candidate.is_file():
            values = dotenv_values(candidate)
            for name in missing:
                value = values.get(name)
                if isinstance(value, str) and value.strip():
                    source[name] = value.strip()
            selected_path = candidate

    missing = [name for name in REQUIRED_SECRET_NAMES if not source.get(name, "").strip()]
    if missing:
        raise ConfigurationError("Missing required environment variable(s): " + ", ".join(missing))

    return Phase3Config(
        nebius_api_key=source["NEBIUS_API_KEY"].strip(),
        pinecone_api_key=source["PINECONE_API_KEY"].strip(),
        secrets_path=selected_path,
    )
