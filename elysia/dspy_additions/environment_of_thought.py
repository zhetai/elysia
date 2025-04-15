import os

import dspy
from dspy.primitives.program import Module
from dspy.signatures.signature import ensure_signature
from pydantic import BaseModel

TRAIN = os.environ.get("TRAIN", "False") == "True"
TRAIN_CONFIRM = os.environ.get("TRAIN_CONFIRM", "False") == "True"


class EnvironmentOfThought(Module):
    def __init__(self, signature, activated=True, message_update=True, **config):
        super().__init__()

        self.activated = activated

        self.signature = signature = ensure_signature(signature)
        *_keys, last_key = signature.output_fields.keys()

        # reasoning style prompts
        reasoning_desc = "Reasoning: Repeat relevant parts of the environment and context, use this to think step by step in order to answer the query."
        # reasoning_desc += ". But only keep a minimum draft for each thinking step, with 5 words at most."  # chain of draft - experimenting TODO: does this butcher outputs in tree?
        reasoning_prefix = "${reasoning}"
        reasoning_field = dspy.OutputField(prefix=reasoning_prefix, desc=reasoning_desc)

        # chat response prompt
        message_update_desc = """
        Continue your current message to the user (latest assistant field in conversation history) with ONE concise sentence that:
        - Describes NEW technical details about your latest action
        - Highlights specific parameters or logic you just applied
        - Avoids repeating anything from conversation history
        - Speaks directly to them (no 'the user'), gender neutral message
        Just provide the new sentence update, not the full message from the conversation history.
        """.strip()
        message_update_prefix = "${message_update}"
        message_update_field: str = dspy.OutputField(
            prefix=message_update_prefix, desc=message_update_desc
        )

        # impossible prompt
        impossible_desc = """
        Given the actions you have available, and the environment/information. Is the task impossible to complete?
        I.e., do you wish that you had a different task to perform/choose from and hence should return to the base of the decision tree?
        """.strip()
        impossible_prefix = "${impossible}"
        impossible_field = dspy.OutputField(
            prefix=impossible_prefix, desc=impossible_desc
        )

        # add to signature
        extended_signature = signature.prepend("reasoning", reasoning_field, type_=str)
        extended_signature = extended_signature.append(
            "impossible", impossible_field, type_=bool
        )
        if message_update:
            extended_signature = extended_signature.append(
                "message_update", message_update_field, type_=str
            )

        self._predict = dspy.Predict(extended_signature, **config)
        self._predict.extended_signature = extended_signature

    def forward(self, **kwargs):
        assert self.activated in [True, False]

        signature = kwargs.pop(
            "new_signature",
            self._predict.extended_signature if self.activated else self.signature,
        )
        return self._predict(signature=signature, **kwargs)

    @property
    def demos(self):
        return self._predict.demos

    @property
    def extended_signature(self):
        return self._predict.extended_signature
