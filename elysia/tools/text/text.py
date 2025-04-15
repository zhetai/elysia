# Prompt Executors
#
import dspy
import dspy.predict

from elysia.dspy_additions.environment_of_thought import EnvironmentOfThought
from elysia.util.reference import create_reference

# LLM
from elysia.objects import Response, Tool
from elysia.tools.text.objects import Summary
from elysia.tools.text.prompt_templates import SummarizingPrompt, TextResponsePrompt

# Objects
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager


class Summarizer(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="summarize",
            description="""
            Summarize retrieved information for the user when all relevant data has been gathered.
            Provides a text response, and may end the conversation, but unlike text_response tool, can be used mid-conversation.
            Avoid for general questions where text_response is available.
            Summarisation text is directly displayed to the user.
            Most of the time, you can choose end_actions to be True to end the conversation with a summary.
            This is a good way to end the conversation.
            """,
            status="Summarizing...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs
    ):
        summarizer = EnvironmentOfThought(SummarizingPrompt, message_update=False)
        summarizer = dspy.asyncify(summarizer)

        summary = await summarizer(
            user_prompt=tree_data.user_prompt,
            reference=create_reference(),
            environment=tree_data.environment.to_json(),
            conversation_history=tree_data.conversation_history,
            lm=base_lm,
        )

        yield Summary(text=summary.summary, title=summary.subtitle)


class TextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="text_response",
            description="""
            One way to end the conversation.
            Can use when no other tools are suitable.
            This should be used when the user has finished their query, or you have nothing more to do except reply.
            """,
            status="Writing response...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs
    ):
        text_response = dspy.Predict(TextResponsePrompt)
        text_response = dspy.asyncify(text_response)

        output = await text_response(
            user_prompt=tree_data.user_prompt,
            reference=create_reference(),
            environment=tree_data.environment.to_json(),
            conversation_history=tree_data.conversation_history,
            current_message=tree_data.current_message,
            lm=base_lm,
        )

        yield Response(text=output.response)


class FakeTextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="text_response",
            description="""
            End the conversation. This should be used when the user has finished their query, or you have nothing more to do except reply.
            """,
            status="Writing response...",
            inputs={
                "text": {
                    "type": "string",
                    "description": (
                        "The text to display to the user. Speak directly to them, do not say 'the user' or similar. "
                        "If you have achieved the goal, give a satsifying answer to their original prompt (user_prompt). "
                        "If you have not achieved the goal, explain. "
                        "If you know why, explain any shortcomings, maybe it was a limitation, a misunderstanding, a problem with the data. "
                        "Your suggestions should be useful to the user so they know if there is something possible to fix or do as a follow up. "
                        "Sometimes you can ask for clarification. "
                        "Be polite, professional, apologetic if necessary but above all - helpful!"
                    ),
                    "default": "",
                }
            },
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs
    ):
        yield Response(text=inputs["text"])
