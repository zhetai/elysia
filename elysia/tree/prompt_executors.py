from typing import Dict, List, Optional

import dspy

# dspy
from elysia.dspy_additions.environment_of_thought import EnvironmentOfThought

# Globals
from elysia.util.reference import create_reference

# Prompt Templates
from elysia.tree.prompt_templates import construct_decision_prompt


class DecisionExecutor(dspy.Module):
    def __init__(self, available_tasks: Optional[List[Dict]] = None, **kwargs):
        self.router = EnvironmentOfThought(construct_decision_prompt(available_tasks))

    def forward(self, **kwargs) -> dspy.Prediction:
        available_actions: dict = kwargs.get("available_actions")
        for action in available_actions:
            if available_actions[action]["inputs"] == {}:
                available_actions[action][
                    "inputs"
                ] = "No inputs are needed for this function."

        decision = self.router(
            user_prompt=kwargs.get("user_prompt"),
            instruction=kwargs.get("instruction"),
            style=kwargs.get("style"),
            agent_description=kwargs.get("agent_description"),
            end_goal=kwargs.get("end_goal"),
            reference=create_reference(),
            conversation_history=kwargs.get("conversation_history"),
            collection_information=kwargs.get("collection_information"),
            tree_count=kwargs.get("tree_count"),
            tasks_completed=kwargs.get("tasks_completed"),
            available_actions=kwargs.get("available_actions"),
            environment=kwargs.get("environment"),
            lm=kwargs.get("lm"),
        )

        return decision
