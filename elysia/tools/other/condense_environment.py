import asyncio

import dspy
import spacy
from rich import print

from elysia.config import nlp
from elysia.util.reference import create_reference
from elysia.objects import Result, Status, Tool
from elysia.tools.other.prompt_executors import CondenseEnvironmentExecutor
from elysia.tools.text.objects import Response
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager


class CondenseEnvironment(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="condense_environment",
            description="Condense the environment into a smaller representation.",
            status="Condensing environment...",
            rule=True,
        )
        self.executor = CondenseEnvironmentExecutor()
        self.token_limit = 20000
        self.num_tokens = 0

    def rule_call(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        # Count number of tokens in the environment
        str_environment = str(tree_data.environment.environment)

        # Process text in batches to avoid memory issues
        batch_size = 1000000  # 1MB text chunks
        num_tokens = 0
        for i in range(0, len(str_environment), batch_size):
            batch = str_environment[i : i + batch_size]
            doc = nlp(batch)
            num_tokens += len(doc)

        if num_tokens <= self.num_tokens:
            # nothing has changed since we checked last time
            criteria_met = False
        else:
            criteria_met = num_tokens > self.token_limit
            self.num_tokens = num_tokens

        return criteria_met

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        with dspy.context(lm=complex_lm):
            condensed_prediction = self.executor(
                user_prompt=tree_data.user_prompt,
                reference=create_reference(),
                conversation_history=tree_data.conversation_history,
                tasks_completed=tree_data.tasks_completed,
                environment=tree_data.environment.environment,
            )

        yield Response(condensed_prediction.reasoning_update_message)
        self.keep_all = condensed_prediction.keep_all
        if not condensed_prediction.keep_all or not condensed_prediction.impossible:
            tree_data.environment.environment = (
                condensed_prediction.condensed_environment
            )
