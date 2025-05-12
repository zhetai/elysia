import dspy
from typing import List
from elysia.tools.text.objects import TextWithCitation


class CitedSummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects from the environment, summarize the information in the objects to answer the user's prompt.
    You will do so by providing a full summary, via a collection of TextWithCitation objects.
    The combination of these 'text' fields in these objects will form the full summary (combined via string concatenation).
    """

    subtitle = dspy.OutputField(description="A subtitle for the summary")
    cited_text: List[TextWithCitation] = dspy.OutputField(
        description="""
        The list of TextWithCitation objects, which will form the full summary when the 'text' fields are concatenated.
        Use markdown formatting in the 'text' fields to break down the information into sections, bullet points, etc to make it easier to read.
        Your summary should take account what the user prompt is, and the information in the environment.
        Your summary should be parsing the retrieved objects, and summarising the information in them that is relevant to the user's prompt.
        For example, picking the most relevant information from the retrieved objects, and discussing it, repeating ONLY relevant information if necessary.
        You should provide useful analysis.

        Make sure that your citations are accurate.
        You can provide cited_text fields which MUST come from the environment.
        Do not include any citations which are not in the environment.

        Take the _REF_ID from items in the environment, only use them as the ref_ids in the TextWithCitation objects.
        Do not use the _REF_ID anywhere in the 'text' fields, they should only be copied as the ref_id in the TextWithCitation objects.

        Ensure that you use newline characters (\\n) in each 'text' field if you are using markdown formatting such as lists, or other formatting.
        """.strip()
    )


class SummarizingPrompt(dspy.Signature):
    """
    Given a user_prompt, as well as a list of retrieved objects, summarize the information in the objects to answer the user's prompt.
    Information about you:
    - You are a chatbot for an app named Elysia.
    - You are a helpful assistant designed to be used in a chat interface and respond to user's prompts in a helpful, friendly, and polite manner.
    - Your primary task is to summarize the information in the retrieved objects to answer the user's prompt.
    Do not list any of the retrieved objects in your response. Do not give an itemised list of the objects, since they will be displayed to the user anyway.
    """

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
