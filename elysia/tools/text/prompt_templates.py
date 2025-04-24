import dspy


class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to summarize the information in the retrieved objects to answer the user's prompt.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    """

    user_prompt = dspy.InputField(description="The user's original query")
    reference = dspy.InputField(
        description="Information about the state of the world NOW such as the date and time, used to frame the summarization.",
        format=str,
    )

    conversation_history = dspy.InputField(
        description="""
        The conversation history between the user and the assistant (you), including all previous messages.
        During this conversation, the assistant has also generated some information, which is also relevant to the decision.
        This information is stored in `environment` field.
        If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses should use these as context.
        The history is a list of dictionaries of the format:
        [
            {
                "role": "user" or "assistant",
                "content": The message
            }
        ]
        In the order which the messages were sent.
        """.strip(),
        format=str,
    )
    environment = dspy.InputField(
        description="""
        The retrieved objects from the knowledge base.
        This will be in the form of a list of dictionaries, where each dictionary contains the metadata and object fields.
        You should use all of the information available to you to answer the user's prompt,
        but use judgement to decide which objects are most relevant to the user's query.
        This has already been displayed to the user, so do not repeat the same information.
        """.strip()
    )
    subtitle = dspy.OutputField(description="A subtitle for the summary")
    summary = dspy.OutputField(
        description="""
        The summary of the retrieved objects. You can use markdown formatting.
        Don't provide an itemised list of the objects, since they will be displayed to the user anyway.
        Your summary should take account what the user prompt is, and the information in the retrieved objects.
        DO NOT list the retrieved objects in your summary.
        Your summary should be parsing the retrieved objects, and summarising the information in them that is relevant to the user's prompt.
        For example, picking the most relevant information from the retrieved objects, and discussing it, repeating ONLY relevant information if necessary.
        """.strip()
    )


class TextResponsePrompt(dspy.Signature):
    """
    You are a helpful assistant, designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    Given a user_prompt, as well as a list of retrieved objects, respond to the user's prompt.
    Your response should be informal, polite, and assistant-like.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to respond to the user's query.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    """

    user_prompt = dspy.InputField(description="The user's original query")
    reference = dspy.InputField(
        description="Information about the state of the world NOW such as the date and time, used to frame the response.",
        format=str,
    )
    conversation_history = dspy.InputField(
        description="""
        The conversation history between the user and the assistant (you), including all previous messages.
        During this conversation, the assistant has also generated some information, which is also relevant to the decision.
        This information is stored in `environment` field.
        If this is non-empty, then you have already been speaking to the user, and these were your responses, so future responses should use these as context.
        The history is a list of dictionaries of the format:
        [
            {
                "role": "user" or "assistant",
                "content": The message
            }
        ]
        In the order which the messages were sent.
        """.strip(),
        format=str,
    )
    environment = dspy.InputField(
        description="""
        The retrieved objects from the knowledge base.
        This will be in the form of a list of dictionaries, where each dictionary contains the metadata and object fields.
        You should use all of the information available to you to answer the user's prompt,
        but use judgement to decide which objects are most relevant to the user's query.
        """.strip()
    )
    response = dspy.OutputField(
        description="""
        The response to the user's prompt.
        If you are explaining how something went wrong, or you could not complete a task, suggest a brief reason why.
        You can suggest alternative questions for the user to ask, alternative ways of phrasing the prompt, or explain your own limitations and how to avoid them.
        E.g., you could ask the user to be more specific, or to provide more information, or explain that you are limited by your data. Be creative and adaptive based on the user's prompt and the type of issue that occurred.
        Use present tense in your text, as if you are currently completing the action.
        Use gender neutral language.
        """.strip()
    )
