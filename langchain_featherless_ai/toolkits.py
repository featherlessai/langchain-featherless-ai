"""FeatherlessAi toolkits."""

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit


class FeatherlessAiToolkit(BaseToolkit):
    # TODO: Replace all TODOs in docstring. See example docstring:
    # https://github.com/langchain-ai/langchain/blob/c123cb2b304f52ab65db4714eeec46af69a861ec/libs/community/langchain_community/agent_toolkits/sql/toolkit.py#L19
    """FeatherlessAi toolkit.

    # TODO: Replace with relevant packages, env vars, etc.
    Setup:
        Install ``langchain-featherless-ai`` and set environment variable ``FEATHERLESSAI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-featherless-ai
            export FEATHERLESSAI_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args:
        arg 1: type
            description
        arg 2: type
            description

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain-featherless-ai import FeatherlessAiToolkit

            toolkit = FeatherlessAiToolkit(
                # ...
            )

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        .. code-block:: none

            # TODO: Example output.

    Use within an agent:
        .. code-block:: python

            from langgraph.prebuilt import create_react_agent

            agent_executor = create_react_agent(llm, tools)

            example_query = "..."

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             # TODO: Example output.

    """  # noqa: E501

    # TODO: This method must be implemented to list tools.
    def get_tools(self) -> List[BaseTool]:
        raise NotImplementedError()
