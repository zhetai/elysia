"""
Test generic prompts and basic outputs.
Such as whether a prompt that requires searching actually does call 'query'.
Same for aggregate, text_response, summarise, itemised_summaries.
"""

import asyncio
import json
import unittest

from deepeval import evaluate, metrics
from deepeval.test_case import LLMTestCase

from elysia.tests.llm.tester import TestTree


class TestGenericPrompts(TestTree):
    def test_query(self):
        """
        Test the query tool is called for a generic prompt requiring it.
        """
        user_prompt = "What github issues discuss the 'PDF upload' issue?"
        tree = asyncio.run(
            self.run_tree(
                user_prompt,
                [],
            )
        )

        collection_names_to_test = [c.lower() for c in tree.tree_data.collection_names]

        # passing empty collection list should mean all collections are searched
        self.assertIn("example_verba_github_issues", collection_names_to_test)
        self.assertIn("example_verba_email_chains", collection_names_to_test)
        self.assertIn("example_verba_slack_conversations", collection_names_to_test)
        self.assertIn("weather", collection_names_to_test)
        self.assertIn("ml_wikipedia", collection_names_to_test)
        self.assertIn("weaviate_blogs", collection_names_to_test)
        self.assertIn("weaviate_documentation", collection_names_to_test)

        self.assertIn("query", tree.decision_history)

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

    def test_aggregate(self):
        """
        Test the aggregate tool is called for a generic prompt requiring it.
        """
        user_prompt = (
            "What is the average price of trousers in the ecommerce collection?"
        )
        tree = asyncio.run(
            self.run_tree(
                user_prompt,
                [
                    "ecommerce",
                    "example_verba_email_chains",
                    "example_verba_github_issues",
                    "example_verba_slack_conversations",
                    "weather",
                    "ML_Wikipedia",
                    "weaviate_blogs",
                    "weaviate_documentation",
                ],
            )
        )

        self.assertIn("aggregate", tree.decision_history)

        # Test with deepeval
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

    def test_text_response(self):
        """
        Test the text_response tool is called for a generic conversational prompt.
        """
        user_prompt = "Hey Elly what's up? What can you do?"
        tree = asyncio.run(
            self.run_tree(
                user_prompt,
                [
                    "Ecommerce",
                    "Example_verba_email_chains",
                    "Example_verba_github_issues",
                    "Example_verba_slack_conversations",
                    "Weather",
                    "ML_Wikipedia",
                    "Weaviate_blogs",
                    "Weaviate_documentation",
                ],
            )
        )

        self.assertIn("text_response", tree.decision_history)

        # Test with deepeval
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

    def test_summarising_output(self):
        """
        Test the summarise tool is called for a generic prompt requiring it.
        """
        user_prompt = (
            "Summarise the 10 most recent issues in the github issues collection"
        )
        tree = asyncio.run(
            self.run_tree(
                user_prompt,
                [
                    "Example_verba_github_issues",
                ],
            )
        )

        self.assertIn("query", tree.decision_history)
        self.assertIn(
            "Example_verba_github_issues",
            tree.tree_data.environment.environment["query"],
        )
        self.assertTrue(
            "summarize" in tree.decision_history
            or tree.decision_history[-1] == "text_response"
        )

        task_metric = metrics.TaskCompletionMetric(
            threshold=0.5, model="gpt-4o-mini", include_reason=True
        )

        tools_called = self.get_tools_called(tree, user_prompt)

        task_test_case = LLMTestCase(
            input=user_prompt,
            actual_output=tree.tree_data.conversation_history[-1]["content"],
            tools_called=tools_called,
        )

        res = evaluate(test_cases=[task_test_case], metrics=[task_metric])
        for test_case in res.test_results:
            self.assertTrue(test_case.success)

        environment = [
            json.dumps(obj)
            for obj in tree.tree_data.environment.environment["query"][
                "Example_verba_github_issues"
            ][0]["objects"]
        ]

        summarisation_metric = metrics.AnswerRelevancyMetric(
            threshold=0.5, model="gpt-4o-mini", include_reason=True
        )
        summarisation_test_case = LLMTestCase(
            input=user_prompt,
            actual_output=tree.tree_data.conversation_history[-1]["content"],
            retrieval_context=environment,
        )

        res = evaluate(
            test_cases=[summarisation_test_case], metrics=[summarisation_metric]
        )
        for test_case in res.test_results:
            self.assertTrue(test_case.success)

    def test_itemised_summaries(self):
        """
        Test the itemised_summaries tool is called when specifically asked for it.
        """
        user_prompt = (
            "Write itemised summaries for each of the 5 most recent message in slack"
        )
        tree = asyncio.run(
            self.run_tree(
                user_prompt,
                [
                    "Example_verba_slack_conversations",
                    "Example_verba_email_chains",
                    "Example_verba_github_issues",
                ],
            )
        )
        self.assertIn("query", tree.decision_history)
        self.assertIn(
            "Example_verba_slack_conversations",
            tree.tree_data.environment.environment["query"],
        )
        for item in tree.tree_data.environment.environment["query"][
            "Example_verba_slack_conversations"
        ][0]["objects"]:
            self.assertTrue(
                "ELYSIA_SUMMARY" in item.keys() and len(item["ELYSIA_SUMMARY"]) > 0
            )


if __name__ == "__main__":
    # unittest.main()

    test = TestGenericPrompts()
    test.test_itemised_summaries()
