import dspy

from elysia.tools.other.prompt_templates import EnvironmentCondenserPrompt


class CondenseEnvironmentExecutor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.condense_environment_prompt = dspy.ChainOfThought(
            EnvironmentCondenserPrompt
        )

    def forward(
        self,
        user_prompt: str,
        reference: dict,
        conversation_history: list[dict],
        tasks_completed: str,
        environment: dict,
    ):
        prediction = self.condense_environment_prompt(
            user_prompt=user_prompt,
            reference=reference,
            conversation_history=conversation_history,
            tasks_completed=tasks_completed,
            environment=environment,
        )
        return prediction
