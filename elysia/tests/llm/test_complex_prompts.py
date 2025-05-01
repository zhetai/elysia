"""
Test more complex prompts that require multiple logical steps.
E.g. aggregation _and_ query.
Focusing only on whether the correct tools are called, not the accuracy of the results.
"""

import asyncio
import json
import os
import unittest

from deepeval import evaluate, metrics
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from elysia.tests.llm.deepeval_setup import CustomLlama
from elysia.tests.llm.tester import TestTree
from elysia import configure

configure(logging_level="DEBUG")


class TestComplexPrompts(TestTree):
    def test_aggregation_and_query(self):
        """
        Test that the correct tools are called for a prompt that requires both aggregation and query.
        """
        user_prompt = "What was the maximum weather in 2015? Then, what was the weather like in the surrounding week of that maximum?"
        tree = asyncio.run(self.run_tree(user_prompt, ["Weather"]))
        self.assertIn("aggregate", tree.decision_history)
        self.assertIn("query", tree.decision_history)

        # -- DeepEval tests --
        deepeval_metric = metrics.TaskCompletionMetric(
            threshold=0.5, model="gpt-4o-mini", include_reason=True
        )

        tools_called = self.get_tools_called(tree, user_prompt)

        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=tree.tree_data.conversation_history[-1]["content"],
            tools_called=tools_called,
        )

        res = evaluate(test_cases=[test_case], metrics=[deepeval_metric])
        for test_case in res.test_results:
            self.assertTrue(test_case.success)

    def test_query_with_chunking(self):
        """
        Test that the correct tools are called for a prompt that requires both aggregation and query.
        """
        user_prompt = "What is HNSW?"
        tree = asyncio.run(
            self.run_tree(user_prompt, ["Weaviate_documentation", "Weaviate_blogs"])
        )
        self.assertIn("query", tree.decision_history)
        self.assertIn("query", tree.tree_data.environment.environment)

        collections_queried = list(
            tree.tree_data.environment.environment["query"].keys()
        )

        for collection in collections_queried:
            for item in tree.tree_data.environment.environment["query"][collection][0][
                "objects"
            ]:
                self.assertTrue("chunk_spans" in item and item["chunk_spans"] is not [])

        # -- DeepEval tests --
        deepeval_metric = metrics.TaskCompletionMetric(
            threshold=0.5, model="gpt-4o-mini", include_reason=True
        )

        tools_called = self.get_tools_called(tree, user_prompt)

        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=tree.tree_data.conversation_history[-1]["content"],
            tools_called=tools_called,
        )

        res = evaluate(test_cases=[test_case], metrics=[deepeval_metric])
        for test_case in res.test_results:
            self.assertTrue(test_case.success)

    def test_impossible_query(self):
        """
        Test that when given a task that is not possible, the model does not attempt it (too many times).
        """
        user_prompt = "What is the weather in January 2025?"
        tree = asyncio.run(self.run_tree(user_prompt, ["Weather"]))

        metric = metrics.GEval(
            name="Judge of agent knowledge lacking awareness",
            criteria="""
            The model should not attempt to answer a question that is not possible to answer.
            So the model may attempt to answer this but should eventually realise it is not possible.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.TOOLS_CALLED,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
        )

        test_case = LLMTestCase(
            input=user_prompt,
            actual_output=tree.tree_data.conversation_history[-1]["content"],
            tools_called=self.get_tools_called(tree, user_prompt),
            expected_output="""
            January 2025 is not a valid date in the weather dataset, only 2014-2016 is available.
            So the model should eventually realise the data is not available.
            """,
        )

        res = evaluate(test_cases=[test_case], metrics=[metric])
        for test_case in res.test_results:
            self.assertTrue(test_case.success)


if __name__ == "__main__":
    # unittest.main()
    test = TestComplexPrompts()
    test.test_query_with_chunking()
