# import asyncio
# import os
# import random
# import time

# import dspy
# from fastapi import APIRouter, Depends
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel

# from elysia.api.dependencies.common import get_user_manager
# from elysia.api.services.client import ClientManager
# from elysia.api.services.user import UserManager
# from elysia.tree.objects import CollectionData, TreeData
# from elysia.tree.prompt_executors import DecisionExecutor
# from elysia.tree.tree import DecisionNode

# dspy.settings.configure(lm=dspy.LM(model="gpt-4o-mini"))

# router = APIRouter()


# class TestConcurrentData(BaseModel):
#     user_prompt: str


# class DecisionNodeTest:
#     def __init__(
#         self,
#         user_prompt: str,
#         instruction: str = "",
#         options: list[dict[str, str]] = [],
#         reference: dict[str, str] = {},
#         conversation_history: list[dict[str, str]] = [],
#         collection_information: dict[str, str] = {},
#         tree_count: str = "",
#         tasks_completed: list[dict[str, str]] = [],
#         current_message: str = "",
#         available_actions: dict[str, str] = {},
#         environment: dict[str, str] = {},
#     ):
#         self.user_prompt = user_prompt
#         self.instruction = instruction
#         self.options = options
#         self.reference = reference
#         self.conversation_history = conversation_history
#         self.collection_information = collection_information
#         self.tree_count = tree_count
#         self.tasks_completed = tasks_completed
#         self.current_message = current_message
#         self.available_actions = available_actions
#         self.environment = environment

#     async def __call__(self):
#         decision_executor = asyncify(DecisionExecutor(self.options))
#         output = await decision_executor(
#             user_prompt=self.user_prompt,
#             instruction=self.instruction,
#             reference=self.reference,
#             conversation_history=self.conversation_history,
#             collection_information=self.collection_information,
#             tree_count=self.tree_count,
#             tasks_completed=self.tasks_completed,
#             current_message=self.current_message,
#             available_actions=self.available_actions,
#             environment=self.environment,
#         )


# @router.post("/test_concurrent")
# async def test_concurrent(test_concurrent_data: TestConcurrentData):
#     user_prompt = test_concurrent_data.user_prompt
#     t = time.time()

#     # Use the asynchronous factory method to create a DecisionNode instance
#     node = DecisionNode(
#         id="1",
#         instruction="Choose one of the following options: [A, B, C], based on what the user says.",
#         options={
#             "A": {
#                 "description": "this is option A",
#                 "future": "",
#                 "inputs": {},
#                 "action": None,
#                 "end": True,
#                 "status": "Running A...",
#                 "next": None,
#                 "rule": False,
#             },
#             "B": {
#                 "description": "this is option B",
#                 "future": "",
#                 "inputs": {},
#                 "action": None,
#                 "end": True,
#                 "status": "Running B...",
#                 "next": None,
#                 "rule": False,
#             },
#             "C": {
#                 "description": "this is option C",
#                 "future": "",
#                 "inputs": {},
#                 "action": None,
#                 "end": True,
#                 "status": "Running C...",
#                 "next": None,
#                 "rule": False,
#             },
#         },
#         root=True,
#         base_lm=dspy.LM(model="gpt-4o-mini"),
#         complex_lm=dspy.LM(model="gpt-4o-mini"),
#     )

#     output, _, _ = await node.decide(
#         tree_data=TreeData(user_prompt=user_prompt),
#         decision_data=DecisionData(
#             instruction="Choose one of the following options: [A, B, C], based on what the user says."
#         ),
#         action_data=ActionData(
#             collection_information=CollectionData(
#                 collection_names=["example_verba_github_issues"], client_manager=None
#             )
#         ),
#     )
#     tend = time.time()

#     return JSONResponse(
#         content={
#             "message": output.full_chat_response,
#             "time_taken": tend - t,
#             "confidence_score": output.confidence_score,
#         }
#     )
#     # decision = asyncify(DecisionExecutor())
#     # output = await decision(
#     #     user_prompt=user_prompt,
#     #     instruction="Choose one of the following options: [A, B, C], based on what the user says.",
#     #     reference={},
#     #     conversation_history=[],
#     #     collection_information={},
#     #     previous_reasoning=[],
#     #     tree_count="0",
#     #     tasks_completed=[],
#     #     current_message="",
#     #     available_actions={},
#     #     environment={}
#     # )

#     # # await asyncio.sleep(5)
#     # tend = time.time()
#     # return JSONResponse(content={"message": output.full_chat_response, "time_taken": tend - t})
