import dspy

from elysia.dspy_additions.environment_of_thought import EnvironmentOfThought
from elysia.tools.other.prompt_templates import EnvironmentCondenserPrompt


class CondenseEnvironmentExecutor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.condense_environment_prompt = EnvironmentOfThought(
            EnvironmentCondenserPrompt
        )

    def forward(
        self,
        user_prompt: str,
        reference: dict,
        conversation_history: list[dict],
        tasks_completed: str,
        current_message: str,
        environment: dict,
    ):
        prediction = self.condense_environment_prompt(
            user_prompt=user_prompt,
            reference=reference,
            conversation_history=conversation_history,
            tasks_completed=tasks_completed,
            current_message=current_message,
            environment=environment,
        )
        return prediction
