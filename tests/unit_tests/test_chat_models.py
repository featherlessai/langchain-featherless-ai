"""Test chat model integration."""

from typing import Type

from langchain_featherless_ai.chat_models import ChatFeatherlessAi
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatFeatherlessAiUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatFeatherlessAi]:
        return ChatFeatherlessAi

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "FEATHERLESSAI_API_KEY": "featherless_api_key",
                "FEATHERLESSAI_API_BASE": "featherless_api_base",
            },
            {
            },
            {
                "featherless_api_key": "featherless_api_key",
                "featherless_api_base": "featherless_api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "featherless-ai/Qwerky-72B",
            "featherless_api_key": "test_api_key",
            "featherless_api_base": "http://localhost:8080/test",
        }