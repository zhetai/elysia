import dspy


class EnvironmentCondenserPrompt(dspy.Signature):
    """
    You are part of an ensemble of models that are answering a user's question.
    However, your task is NOT to answer the user's question.
    Instead, you are given the environment (a collection of items) that is deemed too big to be processed by the other models.

    You are helping to filter this collection of items to keep only the most relevant ones.
    To do so, you will reconstruct the environment with only the most relevant items.
    Know that your output can not be too long, as it will be passed to the next model in the ensemble.
    You are only to keep the most relevant items, and remove the least relevant items.
    Be pragmatic.

    You are NOT designed to answer the user's question.
    But use the user prompt, conversation history and other context to determine what information is most relevant to keep.

    The user is not aware you are completing this filtering task, so if/when you update the user via reasoning_update_message,
    you need to mention what exactly you are doing (within the <NEW> tags, as this is what the user will see).
    """

    # Input fields for context
    user_prompt: str = dspy.InputField(
        description="""
        [FOR CONTEXT]
        The user's original question/prompt that needs to be answered
        """.strip()
    )
    reference: dict = dspy.InputField(
        description="""
        [FOR CONTEXT]
        Current context information (e.g., date, time) for decision-making
        """.strip()
    )
    conversation_history: list[dict] = dspy.InputField(
        description="""
        [FOR CONTEXT]
        Previous messages between user and assistant in chronological order:
        [{"role": "user"|"assistant", "content": str}]
        Use this to maintain conversation context and avoid repetition.
        """.strip()
    )
    tasks_completed: str = dspy.InputField(
        description="""
        [FOR CONTEXT]
        Which tasks have been completed in order. These are numbered so that higher numbers are more recent.
        Separated by prompts (so you should identify the prompt you are currently working on to see what tasks have been completed so far)
        Also includes reasoning for each task, to continue a decision logic across tasks.
        Use this to determine whether future searches, for this prompt are necessary, and what task(s) to choose.
        """.strip()
    )

    environment: str = dspy.InputField(
        description="""
        [FOR CONDENSING]
        Information gathered from completed tasks.
        Empty if no data has been retrieved yet.
        Use to determine if more information is needed.
        """.strip(),
        format=str,
    )

    # Output fields
    keep_all: bool = dspy.OutputField(
        description="""
        Whether to keep all items in the environment. If True, you should NOT populate the condensed_environment field.
        Just return `{}` for the condensed_environment field.
        """.strip(),
        format=bool,
    )
    condensed_environment: dict = dspy.OutputField(
        description="""
        A reconstruction of the environment with only the most relevant information.
        Use the context to determine what information is most relevant.
        You should aim to keep the most relevant information, and remove the least relevant information.
        You can add summaries to condense multiple items into a single item, but it must have the _exact_ same structure as the original item.
        E.g., the original items are a list of dicts ([{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]), the condensed items should be a list of dicts ([{"name": "John and Jane", "age": 55}]).
        Aim to reduce the environment to as few items as possible. Remove items that are not relevant, keep those that are.
        You MUST format your output as a JSON dictionary.
        You most importantly must keep the same structure as the original environment.
        I.e., the same keys, the same exact string values that index the environment, the same format (nested list, nested dict)
        Whatever you see the _format_ of the environment is, you must keep it for your condensed version.
        """.strip(),
        format=dict,
    )
