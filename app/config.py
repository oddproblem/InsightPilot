# EXTENSION POINT: Add new environment variables here.
# Each field corresponds to an env var (by default, uppercase field name).
# Required fields have no default value — the app will fail to start if missing.
# Optional fields have default values shown inline.
#
# Never read os.environ directly in any other file. All config lives here.

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    # ─── Database ─────────────────────────────────────────────────────────────
    database_url_override: str | None = Field(default=None, validation_alias="database_url")
    postgres_host: str = "localhost"  # POSTGRES_HOST
    postgres_port: int = 5432  # POSTGRES_PORT
    postgres_db: str = "agentdb"  # POSTGRES_DB
    postgres_user: str = "agent"  # POSTGRES_USER
    postgres_password: str  # POSTGRES_PASSWORD — required, no default
    database_pool_size: int = 10  # DATABASE_POOL_SIZE

    # ─── LLM — Direct OpenAI ──────────────────────────────────────────────────
    openai_api_key: str = ""  # OPENAI_API_KEY — used for embeddings + direct LLM
    llm_model: str = "gpt-4o-mini"  # LLM_MODEL
    embedding_model: str = "text-embedding-3-small"  # EMBEDDING_MODEL

    # ─── LLM — OpenRouter (optional) ─────────────────────────────────────────
    # When set, LLM calls route through OpenRouter instead of direct OpenAI.
    # Embeddings always use openai_api_key (OpenRouter doesn't support embeddings).
    # OpenRouter model format: "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet",
    #   "google/gemini-flash-1.5", "meta-llama/llama-3.1-70b-instruct", etc.
    openrouter_api_key: str | None = None  # OPENROUTER_API_KEY
    openrouter_base_url: str = "https://openrouter.ai/api/v1"  # OPENROUTER_BASE_URL
    openrouter_site_url: str = "https://github.com/oddproblem/InsightPilot"
    openrouter_app_name: str = "InsightPilot"

    # ─── Optional web search ──────────────────────────────────────────────────
    tavily_api_key: str | None = None  # TAVILY_API_KEY — optional

    # ─── App ──────────────────────────────────────────────────────────────────
    app_env: str = "development"  # APP_ENV: development | production
    log_level: str = "INFO"  # LOG_LEVEL
    log_format: str = "json"  # LOG_FORMAT: json | text

    # ─── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_settings(self) -> "AppConfig":
        if self.app_env not in ("development", "production"):
            raise ValueError(f"APP_ENV must be 'development' or 'production', got: {self.app_env}")
        if not self.openai_api_key and not self.openrouter_api_key:
            raise ValueError("At least one of OPENAI_API_KEY or OPENROUTER_API_KEY must be set.")
        if not self.openai_api_key:
            import warnings

            warnings.warn(
                "OPENAI_API_KEY is not set. Embeddings will fail unless you provide "
                "an OpenAI-compatible embedding endpoint.",
                stacklevel=2,
            )
        return self

    # ─── Computed properties ──────────────────────────────────────────────────

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def llm_api_key(self) -> str:
        """API key used for LLM chat-completion calls.

        Prefers OpenRouter when OPENROUTER_API_KEY is set; falls back to
        direct OpenAI when only OPENAI_API_KEY is provided.
        """
        return self.openrouter_api_key or self.openai_api_key

    @property
    def llm_base_url(self) -> str | None:
        """Base URL for LLM API calls.  None means use the default OpenAI endpoint."""
        return self.openrouter_base_url if self.openrouter_api_key else None

    @property
    def llm_extra_headers(self) -> dict[str, str]:
        """Extra HTTP headers required by OpenRouter (ignored for direct OpenAI)."""
        if not self.openrouter_api_key:
            return {}
        return {
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_app_name,
        }

    @property
    def embedding_api_key(self) -> str:
        """API key used for embedding generation (always OpenAI-compatible)."""
        return self.openai_api_key

    model_config = {"env_file": ".env", "case_sensitive": False}


# Singleton — imported by everything else. Never instantiated twice.
config = AppConfig()  # type: ignore[call-arg]
