"""Test ChatFeatherlessAi chat model."""

from typing import Type
import pytest
from langchain_featherless_ai.chat_models import ChatFeatherlessAi
from langchain_core.messages import AIMessageChunk, BaseMessageChunk
from langchain_tests.integration_tests import ChatModelIntegrationTests
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from typing import Optional, Type


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatFeatherlessAi]:
        return ChatFeatherlessAi

    @property
    def has_tool_calling(self) -> bool:
        return False

    @property
    def tool_choice_value(self) -> str:
        return "tool_name"

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "featherless-ai/Qwerky-72B"
        }

    @pytest.mark.xfail(reason="Not yet supported.")
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        super().test_usage_metadata_streaming(model)