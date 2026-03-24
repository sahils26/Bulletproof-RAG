"""Configuration system using Pydantic BaseSettings.

Settings are loaded from environment variables and ``.env`` files.
API keys use ``SecretStr`` to prevent accidental logging.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["anthropic", "openai"] = Field(
        default="anthropic", description="Which LLM provider to use."
    )
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Model identifier."
    )
    api_key: SecretStr = Field(
        default=SecretStr(""), description="API key (loaded from env)."
    )
    max_tokens: int = Field(default=4096, description="Max tokens per response.")
    temperature: float = Field(default=0.0, description="Sampling temperature.")

    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")


class VectorStoreConfig(BaseSettings):
    """Vector store configuration."""

    backend: Literal["chromadb"] = Field(
        default="chromadb", description="Vector store backend."
    )
    url: str = Field(
        default="http://localhost:8000", description="Vector store service URL."
    )
    collection_name: str = Field(
        default="default", description="Default collection name."
    )

    model_config = SettingsConfigDict(env_prefix="VECTOR_", extra="ignore")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence-transformers model name."
    )
    batch_size: int = Field(default=32, description="Batch size for embedding.")
    dimensions: int = Field(default=384, description="Embedding vector dimensions.")

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")


class PipelineConfig(BaseSettings):
    """RAG pipeline behaviour configuration."""

    max_iterations: int = Field(
        default=3, description="Max self-correction loops."
    )
    relevance_threshold: float = Field(
        default=3.0, description="Minimum relevance score (1–5) to pass grading."
    )
    enable_hallucination_check: bool = Field(
        default=True, description="Whether to run hallucination checker."
    )

    model_config = SettingsConfigDict(env_prefix="PIPELINE_", extra="ignore")


class AppConfig(BaseSettings):
    """Top-level application config — composes all sub-configs."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
