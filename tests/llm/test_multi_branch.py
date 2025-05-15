import os
import pytest
import asyncio
import dspy
from dspy import LM
from elysia.objects import Result
from copy import deepcopy
from elysia import Tool
from elysia.tree.objects import TreeData
from elysia.config import Settings, configure
from elysia.tree.tree import Tree
from elysia.tools.text.text import TextResponse, CitedSummarizer
from elysia.tools.retrieval.query import Query
from elysia.tools.retrieval.aggregate import Aggregate

try:
    # get_ipython
    os.chdir("../..")
except NameError:
    pass

from elysia.util.client import ClientManager


class TestMultiBranch:

    @pytest.mark.asyncio
    async def test_incorrect_branch_id(self):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_default(),
        )

        # error if branch_id is not found
        with pytest.raises(ValueError):
            tree.add_tool(CitedSummarizer, branch_id="incorrect_branch_id")

        # error if no branch_id is provided
        with pytest.raises(ValueError):
            tree.add_tool(CitedSummarizer)

        # no error if root is True
        tree.add_tool(CitedSummarizer, root=True)

    @pytest.mark.asyncio
    async def test_default_multi_branch(self):

        tree = Tree(
            low_memory=False,
            branch_initialisation="multi_branch",
            settings=Settings.from_default(),
        )

        assert len(tree.decision_nodes) > 1

        async for result in tree.async_run(
            user_prompt="Hello",
            collection_names=["Example_verba_github_issues", "Ecommerce", "Weather"],
        ):
            pass

    @pytest.mark.asyncio
    async def test_new_multi_branch(self):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_default(),
        )

        tree.add_branch(
            "respond_to_user",
            "choose between citing objects from the environment, or not citing any objects",
            "choose when responding to user",
            from_branch_id="base",
        )
        tree.add_branch(
            "search_for_objects",
            "choose between searching for objects from the environment, or aggregating objects from the environment",
            "choose when searching for objects",
            from_branch_id="base",
        )

        tree.add_tool(CitedSummarizer, branch_id="respond_to_user")
        tree.add_tool(TextResponse, branch_id="respond_to_user")
        tree.add_tool(Query, branch_id="search_for_objects")
        tree.add_tool(Aggregate, branch_id="search_for_objects")

        assert len(tree.decision_nodes) == 3

        async for result in tree.async_run(
            user_prompt="What was the most recent github issue?",
            collection_names=["Example_verba_github_issues", "Ecommerce", "Weather"],
        ):
            pass
