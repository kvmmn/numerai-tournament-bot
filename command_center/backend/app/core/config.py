from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


NUMERAI_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "Numerai Command Center"
    API_V1_STR: str = "/api/v1"
    API_ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    NUMERAI_PUBLIC_ID: str | None = None
    NUMERAI_SECRET_KEY: str | None = None

    DATA_VERSION: str = "v5.2"
    DATA_DIR: str = str(NUMERAI_ROOT)

    MODEL_NAME: str = "baseline_lgbm"
    FEATURE_SET: str = "small"
    TRAIN_ERA_STRIDE: int = 4
    MIN_TRAIN_ERAS: int = 100
    NEUTRALIZATION_PROPORTION: float = 0.75
    AUTO_APPROVE_SUBMISSION: bool = False

    # Comma-separated list, e.g. "KVMMN,KVMMN_FN,KVMMN_TE"
    NUMERAI_MODEL_NAMES: str | None = None
    # A daily submission artifact may target exactly one model slot. This avoids
    # silently uploading identical predictions to unrelated strategies.
    SUBMISSION_TARGET_MODEL: str | None = None

    # --- Auto-approval thresholds (full-auto mode) ---
    MIN_VALIDATION_CORR: float = 0.01
    MIN_SHARPE_RATIO: float = 0.3
    MAX_FEATURE_EXPOSURE: float = 0.1
    USE_ENSEMBLE: bool = True  # Use multi-model ensemble by default

    # --- Durable control plane ---
    CONTROL_PLANE_DIR: str = str(BACKEND_ROOT / "automation" / "state")
    MODEL_REGISTRY_DIR: str = str(BACKEND_ROOT / "automation" / "model_registry")
    APPROVAL_TTL_MINUTES: int = 180
    SUBMISSION_MIN_UNIQUE_PREDICTIONS: int = 100

    # --- Staking safety policy ---
    # Stake changes are always human approved, even when submission automation
    # is enabled. Zero disables stake increases until limits are configured.
    MAX_TOTAL_STAKE_NMR: float = 0.0
    MAX_MODEL_STAKE_NMR: float = 0.0
    MAX_STAKE_CHANGE_NMR: float = 0.0

    # --- Competition participation and season monitoring ---
    SEASON_QUALIFYING_ROUNDS: int = 20
    SEASON_MIN_AT_RISK_NMR: float = 1.0
    ACCOUNT_RANK_SCAN_LIMIT: int = 1000

    # Numerai MCP (remote) config
    NUMERAI_MCP_URL: str = "https://api-tournament.numer.ai/mcp/sse"
    # Format expected by Numerai docs: "Token PUBLIC_KEY$PRIVATE_KEY"
    NUMERAI_MCP_AUTH: str | None = None
    NUMERAI_MCP_USE_SSE: bool = True

    # Optional tool mapping for MCP-first workflow
    NUMERAI_MCP_TOOL_CURRENT_ROUND: str | None = None
    NUMERAI_MCP_TOOL_SYNC_DATA: str | None = None
    NUMERAI_MCP_TOOL_TRAIN_MODEL: str | None = None
    NUMERAI_MCP_TOOL_EVALUATE_MODEL: str | None = None
    NUMERAI_MCP_TOOL_SUBMIT_PREDICTIONS: str | None = None

    model_config = SettingsConfigDict(
        env_file=(str(BACKEND_ROOT / ".env"), str(NUMERAI_ROOT / ".env")),
        env_ignore_empty=True,
        extra="ignore",
    )


settings = Settings()
