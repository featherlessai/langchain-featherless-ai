"""Test FeatherlessAi embeddings."""

from typing import Type

from langchain_featherless_ai.embeddings import FeatherlessAiEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[FeatherlessAiEmbeddings]:
        return FeatherlessAiEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
