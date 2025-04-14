import dspy

# Prompt Templates
from elysia.api.services.prompt_templates import (
    FollowUpSuggestionsPrompt,
    InstantReplyPrompt,
    ObjectRelevancePrompt,
    TitleCreatorPrompt,
)

# dspy
from elysia.dspy.environment_of_thought import EnvironmentOfThought


class TitleCreatorExecutor(dspy.Module):
    def __init__(self):
        self.title_creator_prompt = dspy.ChainOfThought(TitleCreatorPrompt)

    def __call__(self, text: str, lm: dspy.LM) -> str:
        return self.title_creator_prompt(text=text, lm=lm)


class ObjectRelevanceExecutor(dspy.Module):
    def __init__(self):
        self.object_relevance_prompt = EnvironmentOfThought(ObjectRelevancePrompt)

    def __call__(self, user_prompt: str, objects: list[dict]) -> bool:
        return self.object_relevance_prompt(user_prompt=user_prompt, objects=objects)


class InstantReplyExecutor(dspy.Module):
    def __init__(self):
        self.instant_reply_prompt = dspy.Predict(InstantReplyPrompt)

    def __call__(self, user_prompt: str) -> str:
        return self.instant_reply_prompt(user_prompt=user_prompt)


class FollowUpSuggestionsExecutor(dspy.Module):
    def __init__(self):
        self.follow_up_suggestions_prompt = dspy.Predict(FollowUpSuggestionsPrompt)

    def __call__(self, **kwargs) -> str:
        pred = self.follow_up_suggestions_prompt(**kwargs)
        # print(pred.reasoning)
        return pred
