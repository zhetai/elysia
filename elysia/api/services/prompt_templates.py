import dspy


class TitleCreatorPrompt(dspy.Signature):
    """
    You are an expert at creating a title for a given text.
    Your goal is not to answer the question or do anything else, just to create a title for what the ensuing conversation is about.
    """

    text = dspy.InputField(
        description="""
        The text to create a title for. This is usually a prompt from the user, which you will use to create a title of a conversation.
        """.strip()
    )
    title = dspy.OutputField(
        description="""
        The title for the text. This is a single, short, succinct summary that describes the topic of the conversation.
        """.strip()
    )


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


class FollowUpSuggestionsPrompt(dspy.Signature):
    """
    Expert at suggesting engaging follow-up questions based on recent user interactions.
    Generate questions that showcase system capabilities and maintain user interest.
    Questions should be fun, creative, and designed to impress.

    System context:
    An agentic RAG service querying or aggregating information from Weaviate collections via filtering, sorting, multi-collection queries, summary statistics.
    Uses tree-based approach with specialized agents and decision nodes for dynamic information retrieval,
    summary generation, and real-time results display while maintaining natural conversation flow.

    Since you are suggesting _follow-up_ questions, you should not suggest questions that are too similar to the user's prompt.
    Instead, try to think of something that can connect different data sources together, or provide new insights that the user may not have thought of.
    These should be fully formed questions, that you think the system is capable of answering, that the user would be interested in, and that show off the system's capabilities.

    Use `data_information` to see if there are cross-overs between the already retrieved data (environment) and any other data sources.
    It is important you study these in depth so you do not create follow-up questions that would be impossible or very difficult to answer.

    Do not suggest questions that will require _complex_ calculations or reasoning, it should be mostly focused on following up on the already retrieved data,
    or following up by retrieving similar or new data that is related to the already retrieved data.
    """

    user_prompt = dspy.InputField(description="The user's input prompt.")
    reference: dict = dspy.InputField(
        desc="""
        Current state information like date/time to frame.
        """.strip(),
        format=str,
    )
    conversation_history: list[dict] = dspy.InputField(
        description="""
        Full conversation history between user and assistant, including generated information in environment field.
        Format: [{"role": "user"/"assistant", "content": message}]
        Use as context for future responses if non-empty.
        """.strip(),
        format=str,
    )
    environment: str = dspy.InputField(
        description="Information gathered from completed tasks. These are already shown to the user. Use for context, and to cross reference with data_information.",
        format=str,
    )
    data_information: dict = dspy.InputField(
        desc="""
        Collection metadata to inform query choices. Format per collection:
        {
            "name": data collection name,
            "length": object count,
            "summary": collection summary,
            "fields": {
                "field_name": {
                    "groups": unique text values (empty if non-text),
                    "mean": average field length (tokens/list items),
                    "range": min/max length values,
                    "type": data type
                }
            }
        }
        Use this to see if there are cross-overs between the currently retrieved data (environment) and any other collections, to help make an interesting question.
        Also use this to NOT suggest questions that would be impossible.
        This is IMPORTANT. Do not suggest questions that would be impossible, i.e., you can see the distinct categories between the collections, and look for overlap between the environment and these groups.
        If you can't find any overlap, do not assume there is any.
        Instead, suggest a question that would be a generic follow-up based on the environment/prompt.
        But if you do see overlap, suggest a question that would be a follow-up based on the overlap!
        """.strip(),
        format=str,
    )
    old_suggestions: list[str] = dspy.InputField(
        description="A list of questions that have already been suggested. Do not suggest the same question (or anything similar) as any items in this list.",
        format=str,
    )
    suggestions: list[str] = dspy.OutputField(
        description="A list of follow up questions to the user's prompt. Around 2 questions. These should be punctual, short, around 10 words or so. Try to mix up the vocabulary and type of question between them."
    )
