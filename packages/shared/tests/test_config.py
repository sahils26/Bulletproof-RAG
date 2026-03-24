"""Tests for shared config system — defaults and environment overrides."""

import os

from shared.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    PipelineConfig,
    VectorStoreConfig,
)


class TestConfigDefaults:
    def test_llm_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "anthropic"
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.0

    def test_vector_store_defaults(self):
        cfg = VectorStoreConfig()
        assert cfg.backend == "chromadb"
        assert cfg.url == "http://localhost:8000"

    def test_embedding_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model_name == "all-MiniLM-L6-v2"
        assert cfg.dimensions == 384

    def test_pipeline_defaults(self):
        cfg = PipelineConfig()
        assert cfg.max_iterations == 3
        assert cfg.relevance_threshold == 3.0
        assert cfg.enable_hallucination_check is True

    def test_app_config_composes_all(self):
        cfg = AppConfig()
        assert cfg.llm.provider == "anthropic"
        assert cfg.vector_store.backend == "chromadb"
        assert cfg.embedding.model_name == "all-MiniLM-L6-v2"


class TestConfigEnvOverride:
    def test_pipeline_override(self):
        os.environ["PIPELINE_MAX_ITERATIONS"] = "5"
        os.environ["PIPELINE_RELEVANCE_THRESHOLD"] = "4.0"

        cfg = PipelineConfig()
        assert cfg.max_iterations == 5
        assert cfg.relevance_threshold == 4.0

        del os.environ["PIPELINE_MAX_ITERATIONS"]
        del os.environ["PIPELINE_RELEVANCE_THRESHOLD"]

    def test_vector_store_override(self):
        os.environ["VECTOR_BACKEND"] = "chromadb"
        os.environ["VECTOR_URL"] = "http://remote:9000"

        cfg = VectorStoreConfig()
        assert cfg.url == "http://remote:9000"

        del os.environ["VECTOR_BACKEND"]
        del os.environ["VECTOR_URL"]
