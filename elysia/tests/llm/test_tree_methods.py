import os
import unittest

import dspy
import elysia
from elysia.config import Settings, configure
from elysia.config import settings as global_settings
from elysia.tree.tree import Tree

from litellm import completion


class TestTreeMethods(unittest.TestCase):

    def do_query(self, user_prompt: str):
        elysia.config.settings.default_config()
        tree = Tree(
            debug=False,
        )
        tree.run(
            user_prompt,
            collection_names=[
                "Example_verba_github_issues",
                "example_verba_slack_conversations",
                "example_verba_email_chains",
            ],
        )
        return tree

    def test_various_methods(self):

        # set up tree with a basic query
        user_prompt = "What was Edward's last message?"
        tree = self.do_query(user_prompt)

        # check the user prompt is in the conversation history
        self.assertEqual(
            tree.tree_data.conversation_history[0]["content"],
            user_prompt,
        )
        self.assertEqual(tree.tree_data.conversation_history[0]["role"], "user")

        # check some variables are set
        self.assertGreater(tree.tree_data.num_trees_completed, 0)
        self.assertEqual(tree.tree_data.user_prompt, user_prompt)
        self.assertGreater(len(tree.decision_history), 0)
        self.assertIn("query", tree.decision_history)

        # this should have returned something in the environment
        self.assertTrue("query" in tree.tree_data.environment.environment)
        for env_name in tree.tree_data.environment.environment["query"]:
            for returned_object in tree.tree_data.environment.environment["query"][
                env_name
            ]:
                self.assertIn("metadata", returned_object)
                self.assertGreater(len(returned_object["objects"]), 0)

        # this should have updated tasks_completed
        self.assertGreater(len(tree.tree_data.tasks_completed), 0)
        prompt_found = False
        for task in tree.tree_data.tasks_completed:
            if task["prompt"] == user_prompt:
                prompt_found = True
                self.assertGreater(len(task["task"]), 0)
                break

        self.assertTrue(prompt_found)

        # check the tree resets
        tree.soft_reset()

        # These should not change
        self.assertGreater(len(tree.tree_data.conversation_history), 0)
        self.assertGreater(len(tree.tree_data.tasks_completed), 0)
        self.assertTrue(
            "query" in tree.tree_data.environment.environment
        )  # environment should not be reset

        # these should be reset
        self.assertEqual(tree.tree_data.num_trees_completed, 0)
        self.assertEqual(tree.tree_index, 1)
        self.assertEqual(len(tree.decision_history), 0)


if __name__ == "__main__":
    unittest.main()
