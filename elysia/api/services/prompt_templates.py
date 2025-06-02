import dspy


class ObjectRelevancePrompt(dspy.Signature):
    """
    You are an expert at determining the relevance of a set of retrieved objects to a user's query.
    """

    user_prompt = dspy.InputField(
        description="""
        The user's input prompt. This is usually a request for information.
        """.strip()
    )
    objects = dspy.InputField(
        description="""
        The set of retrieved objects.
        These are usually a list of dictionaries, with all dictionaries having the same keys.
        You will have to infer the content of these by inspecting the keys and values.
        """.strip()
    )
    any_relevant: bool = dspy.OutputField(
        description="""
        Whether any of the objects are relevant to the user's query. (True/False)
        Return a boolean value, True or False only, with no other text.
        """.strip()
    )


class InstantReplyPrompt(dspy.Signature):
    """
    You are an expert at generating a quick, concise response to a user's prompt.
    You are part of ensemble of agents that are working together to satisfy the user's request.
    Your job is to generate a response that is a single sentence, and is a quick, concise response to the user_prompt.
    You are NOT tasked with answering the question, just to generate a response that is a single sentence so that the user can be satisfied.
    This is because the full answer will be generated in the next step, and will take a little while to generate.
    So you are just here to make the user feel like the system is responsive.
    """

    user_prompt = dspy.InputField(description="The user's input prompt.")
    response = dspy.OutputField(
        description="""
        The response to the user's prompt.
        This is a single sentence, and is a quick, concise response to the user_prompt.
        Be friendly, concise and polite. Use gender neutral language.
        You should be specific to what the user is asking for (so it is obvious they are listened to - try to repeat back what they said where you can), but without giving any indication that the task required is possible.
        For example, if the user asks "What is the weather in Tokyo?", you should respond with "Let me see if I can check the weather there for you."
        But do not say "I'm checking the weather in Tokyo for you" or anything similar, because it might not be possible.
        Don't give any definitive answers, just that you'll get back to them. Try to not start with "Let me see if" - mix it up, be fun, be creative, be responsive, match the user's tone!
        """.strip()
    )
