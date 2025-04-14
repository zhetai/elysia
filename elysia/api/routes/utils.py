# DSPy
import dspy

# FastAPI
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    DebugData,
    FollowUpSuggestionsData,
    GetUserRequestsData,
    InstantReplyData,
    NERData,
    ObjectRelevanceData,
    TitleData,
)

# Logging
from elysia.api.core.logging import logger

# Dependencies
from elysia.api.dependencies.common import get_user_manager

# LLM
from elysia.api.services.prompt_executors import (
    FollowUpSuggestionsExecutor,
    InstantReplyExecutor,
    ObjectRelevanceExecutor,
    TitleCreatorExecutor,
)

# Services
from elysia.api.services.user import UserManager

# Settings
from elysia.config import nlp

# util
from elysia.util.reference import create_reference
from elysia.util.lm import load_lm

router = APIRouter()

@router.post("/ner")
async def named_entity_recognition(data: NERData):
    """
    Performs Named Entity Recognition using spaCy.
    Returns a list of entities with their labels, start and end positions.
    """
    doc = nlp(data.text)

    out = {"text": data.text, "entity_spans": [], "noun_spans": [], "error": ""}

    try:
        # Get entity spans
        for ent in doc.ents:
            out["entity_spans"].append((ent.start_char, ent.end_char))

        # Get noun spans
        for token in doc:
            if token.pos_ == "NOUN":
                span = doc[token.i : token.i + 1]
                out["noun_spans"].append((span.start_char, span.end_char))

        logger.info(f"Returning NER results: {out}")
    except Exception as e:
        logger.error(f"Error in NER: {str(e)}")
        out["error"] = str(e)

    return JSONResponse(content=out, status_code=200)


@router.post("/title")
async def title(data: TitleData, user_manager: UserManager = Depends(get_user_manager)):

    # check if user has exceeded max requests
    if await user_manager.check_user_exceeds_max_requests(data.user_id):
        return JSONResponse(
            content={"title": "", "error": "User has exceeded max requests"},
            status_code=401,
        )

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    try:
        title_creator = dspy.asyncify(TitleCreatorExecutor())
        title = await title_creator(text=data.text, lm=load_lm())
        logger.info(f"Returning title: {title.title}")
    except Exception as e:
        logger.error(f"Error in title: {str(e)}")
        out = {"title": "", "error": str(e)}
        return JSONResponse(content=out, status_code=200)

    out = {"title": title.title, "error": ""}
    return JSONResponse(content=out, status_code=200)


@router.post("/instant_reply")
async def instant_reply(
    data: InstantReplyData, user_manager: UserManager = Depends(get_user_manager)
):

    # rate limiting
    if await user_manager.check_user_exceeds_max_requests(data.user_id):
        return JSONResponse(
            content={"response": "", "error": "User has exceeded max requests"},
            status_code=401,
        )

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    instant_reply = dspy.asyncify(InstantReplyExecutor())
    with dspy.context(lm=dspy.LM("openrouter/google/gemini-2.0-flash-001")):
        try:
            prediction = await instant_reply(user_prompt=data.user_prompt)
            response = prediction.response

        except Exception as e:
            response = "I'll get back to you in just a moment."
            error = str(e)

    return JSONResponse(content={"response": response, "error": error}, status_code=200)


@router.post("/object_relevance")
async def object_relevance(
    data: ObjectRelevanceData, user_manager: UserManager = Depends(get_user_manager)
):
    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    # rate limiting
    if await user_manager.check_user_exceeds_max_requests(data.user_id):
        return JSONResponse(
            content={"response": "", "error": "User has exceeded max requests"},
            status_code=401,
        )

    error = ""
    object_relevance = ObjectRelevanceExecutor()

    try:
        tree = await user_manager.get_tree(data.user_id, data.conversation_id)
        user_prompt = tree.query_id_to_prompt[data.query_id]
        prediction = object_relevance(user_prompt, data.objects)
        any_relevant = prediction.any_relevant

    except Exception as e:
        any_relevant = True
        error = str(e)

    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "any_relevant": any_relevant,
            "error": error,
        },
        status_code=200,
    )


@router.post("/follow_up_suggestions")
async def follow_up_suggestions(
    data: FollowUpSuggestionsData, user_manager: UserManager = Depends(get_user_manager)
):
    # rate limiting
    if await user_manager.check_user_exceeds_max_requests(data.user_id):
        return JSONResponse(
            content={"response": "", "error": "User has exceeded max requests"},
            status_code=401,
        )

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    # wait for tree event to be completed
    await user_manager.event.wait()

    # get tree from user_id, conversation_id
    tree = await user_manager.get_tree(data.user_id, data.conversation_id)
    error = ""

    # if tree.verbosity > 1:
    #     print(f"[follow_up_suggestions] User ID: {data.user_id}")
    #     print(f"[follow_up_suggestions] Conversation ID: {data.conversation_id}")
    #     print(f"[follow_up_suggestions] User manager ID: {user_manager.manager_id}")
    #     print(f"[follow_up_suggestions] Tree conversation history: {tree.tree_data.conversation_history}")

    follow_up_suggestor = FollowUpSuggestionsExecutor()
    follow_up_suggestor.load(
        "elysia/training/dspy_models/followup_suggestions/fewshot/gemini-2.0-flash-001_gemini-2.0-flash-001.json"
    )
    follow_up_suggestor = dspy.asyncify(follow_up_suggestor)

    with dspy.context(lm=dspy.LM("openrouter/google/gemini-2.0-flash-001")):
        try:
            prediction = await follow_up_suggestor(
                user_prompt=tree.tree_data.user_prompt,
                reference=create_reference(),
                conversation_history=tree.tree_data.conversation_history,
                environment=tree.tree_data.environment.to_json(),
                data_information=tree.tree_data.output_collection_metadata(
                    with_mappings=False
                ),
                old_suggestions=tree.suggestions,
            )
            suggestions = prediction.suggestions
        except Exception as e:
            suggestions = []
            error = str(e)

    tree.suggestions.extend(suggestions)

    return JSONResponse(
        content={"suggestions": suggestions, "error": error}, status_code=200
    )


@router.post("/debug")
async def debug(data: DebugData, user_manager: UserManager = Depends(get_user_manager)):
    tree = await user_manager.get_tree(data.user_id, data.conversation_id)
    if tree.debug:
        base_lm = tree.base_lm
        complex_lm = tree.complex_lm

        histories = [None] * 2
        for i, lm in enumerate([base_lm, complex_lm]):
            histories[i] = []
            for lm_history in lm.history:
                message_thread = []
                for message in lm_history["messages"]:
                    if isinstance(message["content"], list):
                        # assume its a system message list with one dict element
                        message["content"] = message["content"][0]["text"]
                    message_thread.append(message)

                message_thread.append(
                    {
                        "role": "assistant",
                        "content": lm_history["response"].choices[0].message.content,
                    }
                )
                histories[i].append(message_thread)

        out = {
            "base_lm": {"model": base_lm.model, "chat": histories[0]},
            "complex_lm": {"model": complex_lm.model, "chat": histories[1]},
        }
        return JSONResponse(content=out, status_code=200)

    else:
        return JSONResponse(content={}, status_code=200)


@router.post("/get_user_requests")
async def get_user_requests(
    data: GetUserRequestsData, user_manager: UserManager = Depends(get_user_manager)
):
    num_requests, max_requests = await user_manager.get_user_requests(data.user_id)
    return JSONResponse(
        content={"num_requests": num_requests, "max_requests": max_requests},
        status_code=200,
    )
