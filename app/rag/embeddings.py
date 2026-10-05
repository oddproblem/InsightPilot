"""Embeddings generation interface for InsightPilot.

Uses any OpenAI-compatible embeddings API (e.g. direct OpenAI, Azure OpenAI,
or a self-hosted proxy).  OpenRouter does NOT support embeddings — always use
an OpenAI key here even when LLM calls go through OpenRouter.
"""

from openai import OpenAI

from app.config import config


class EmbeddingsService:
    """Service for generating dense vector embeddings via an OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or config.embedding_api_key
        self.model = model or config.embedding_model
        self.base_url = base_url or config.embedding_base_url
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(
                    "No embedding API key configured. "
                    "Set OPENAI_API_KEY or OPENROUTER_API_KEY to enable "
                    "document ingestion and retrieval."
                )
            if self.base_url:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                self._client = OpenAI(api_key=self.api_key)
        return self._client

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a single search query."""
        response = self.client.embeddings.create(
            input=query,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of document texts."""
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]


# Singleton instance
embeddings_service = EmbeddingsService()
