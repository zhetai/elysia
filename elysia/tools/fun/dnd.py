from elysia import Tool
import dspy
import random
from elysia.tools.text.objects import Summary, Text
from elysia.tools import BoringGeneric, Ecommerce
from elysia import Status
from pydantic import BaseModel
from openai import AsyncOpenAI

import asyncio


class Character(BaseModel):
    name: str
    description: str
    appearance: str
    race: str
    character_class: str
    level: int
    personality: str


class Item(BaseModel):
    name: str
    description: str
    type: str
    appearance: str
    rarity: str
    value: int
    weight: int
    is_consumable: bool
    is_equipable: bool
    is_usable: bool


class Descriptor(dspy.Signature):
    """
    You decide the outcome of the user's action in a dungeons and dragons style campaign.
    You are part of an ensemble of AI agents that are crafting and creating a story for the user.
    You are just responsible for ONLY _describing_ what happens in the world as a consequence of the user's action,
    and nothing else.
    """

    user_prompt = dspy.InputField(
        description="What the user has written as their action."
    )
    relevant_context = dspy.InputField(
        description="The current state of the story/world, where the user is, what they are doing, etc."
    )
    relevant_characters: list[Character] = dspy.InputField(
        description="Any characters that have been introduced in the story so far."
    )
    relevant_items: list[Item] = dspy.InputField(
        description="Any items that have been introduced in the story so far."
    )
    consequences: str = dspy.OutputField(
        description="""
        What the user's action will lead to.
        This is a continuation of the story, so it should be a continuation of the current state of the story.
        Pertain to what the user has written in the `user_prompt` field.
        Be poetic and stylistic as a dungeon master would.
        """.strip()
    )
    title: str = dspy.OutputField(
        description="""
        A short title for what the consequences are/story you have just written. Brief.
        """.strip()
    )
    new_characters: list[Character] = dspy.OutputField(
        description="""
        Any new characters that are introduced as a result of what you have written in the `consequences` field.
        Return a list of characters, this should be populated with _any_ character that is introduced in the `consequences` field 
        (and is not already in the `relevant_characters` field).
        """.strip()
    )
    new_items: list[Item] = dspy.OutputField(
        description="""
        Any new items that are introduced as a result of what you have written in the `consequences` field.
        Return a list of items, this should be populated with _any_ item that is introduced in the `consequences` field.
        """.strip()
    )


class DiceRoll(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="roll_dice",
            description="""
            Roll a number of dice with a given number of sides.
            Call this tool when the player tries to make something related to a skill.
            If they are doing something that requires some kind of effort, or some skill, call this tool.
            You can call other tools later to describe the actions of this.
            You decide the skill that the player is trying to use, do not ask the player for it.
            """,
            inputs={
                "skill": {
                    "type": str,
                    "description": "The skill that the player is trying to use",
                    "required": True,
                },
                "num_dice": {
                    "type": int,
                    "description": "The number of dice to roll",
                    "default": 1,
                    "required": False,
                },
                "num_sides": {
                    "type": int,
                    "description": "The number of sides on the dice",
                    "default": 20,
                    "required": False,
                },
                "value_to_exceed": {
                    "type": int,
                    "description": "The value to exceed to succeed",
                    "default": 10,
                    "required": False,
                },
            },
            end=False,
        )

    async def __call__(self, **kwargs):
        num_dice = kwargs.get("num_dice", 1)
        num_sides = kwargs.get("num_sides", 20)
        value_to_exceed = kwargs.get("value_to_exceed", 10)
        roll = random.randint(1, num_sides) * num_dice
        success_message = (
            "The roll was a success!"
            if roll >= value_to_exceed
            else "The roll was a failure!"
        )
        yield BoringGeneric(
            objects=[
                {
                    "title": f"{num_dice}d{num_sides}",
                    "content": f"{roll}",
                    "value_to_exceed": value_to_exceed,
                    "success": roll >= value_to_exceed,
                }
            ],
            metadata={
                "num_dice": num_dice,
                "num_sides": num_sides,
                "roll": roll,
            },
            llm_message=f"Rolled {num_dice}d{num_sides} for the action {kwargs.get('tree_data').user_prompt} and got {roll}. {success_message}",
        )


async def create_characters_and_items(characters: list[Character], items: list[Item]):
    client = AsyncOpenAI()
    images = await asyncio.gather(
        *[
            client.images.generate(
                model="dall-e-3",
                prompt=f"A portrait of {character.name} the {character.character_class} of {character.race} race. "
                f"Description: {character.description}. "
                f"Appearance: {character.appearance}. "
                "Dungeons and dragons, fantasy style.",
                n=1,
                size="1024x1024",
            )
            for character in characters
        ],
        *[
            client.images.generate(
                model="dall-e-3",
                prompt=f"An image of an item, called {item.name}. It is a {item.type} of {item.rarity} rarity. "
                f"Description: {item.description}. "
                f"Appearance: {item.appearance}. "
                "Dungeons and dragons, fantasy style.",
                n=1,
                size="1024x1024",
            )
            for item in items
        ],
    )
    images = [response.data[0].url for response in images]
    character_images = images[: len(characters)]
    item_images = images[len(characters) :]

    return [
        Ecommerce(
            objects=[
                {
                    "name": c.name,
                    "description": f"{c.description}. {c.personality}",
                    "race": c.race,
                    "character_class": c.character_class,
                    "level": c.level,
                    "image": image_url,
                }
                for c, image_url in zip(characters, character_images)
            ],
            metadata={"number_of_new_characters": len(characters)},
            name="characters",
            mapping={
                "name": "name",
                "description": "description",
                "category": "race",
                "subcategory": "character_class",
                "price": "level",
                "image": "image",
            },
            llm_message="Created {number_of_new_characters} characters.",
        ),
        Ecommerce(
            objects=[
                {
                    "name": i.name,
                    "description": i.description,
                    "type": i.type,
                    "rarity": i.rarity,
                    "weight": i.weight,
                    "value": i.value,
                    "is_consumable": i.is_consumable,
                    "is_equipable": i.is_equipable,
                    "is_usable": i.is_usable,
                    "image": image_url,
                }
                for i, image_url in zip(items, item_images)
            ],
            metadata={"number_of_new_items": len(items)},
            name="items",
            mapping={
                "name": "name",
                "description": "description",
                "category": "rarity",
                "subcategory": "type",
                "collection": "weight",
                "price": "value",
                "image": "image",
            },
            llm_message="Created {number_of_new_items} items.",
        ),
    ]


class ActionConsequence(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="action_consequence",
            description="""
            Describe the consequences of the user's action.
            Use this only if the user's action has no more requirements (e.g. dice rolls or confirmations).
            """,
            end=True,
        )
        self.logger = kwargs.get("logger")

    async def __call__(self, **kwargs):
        descriptor = dspy.ChainOfThought(Descriptor)
        descriptor = dspy.asyncify(descriptor)

        tree_data = kwargs.get("tree_data")
        environment = tree_data.environment

        current_characters = []
        current_characters1 = environment.find("set_the_scene", "characters")
        if current_characters1 is not None:
            current_characters += current_characters1

        current_characters2 = environment.find("action_consequence", "characters")
        if current_characters2 is not None:
            current_characters += current_characters2

        current_items = []
        current_items1 = environment.find("set_the_scene", "items")
        if current_items1 is not None:
            current_items += current_items1

        current_items2 = environment.find("action_consequence", "items")
        if current_items2 is not None:
            current_items += current_items2

        current_story1 = environment.find("set_the_scene", "story")
        current_story2 = environment.find("action_consequence", "story")

        current_story = ""
        if current_story1 is not None:
            current_story += "\n".join(
                [c["objects"][0]["text"] for c in current_story1]
            )

        if current_story2 is not None:
            current_story += "\n".join(
                [c["objects"][0]["text"] for c in current_story2]
            )

        yield Status("Writing what happens next...")

        result = await descriptor(
            user_prompt=kwargs.get("tree_data").user_prompt,
            relevant_context=current_story,
            relevant_characters=current_characters,
            lm=kwargs.get("complex_lm"),
        )

        yield Summary(
            text=result.consequences,
            title=result.title,
        )

        yield Status("Creating characters and items...")

        self.logger.info(f"Creating characters: {result.new_characters}")
        self.logger.info(f"Creating items: {result.new_items}")
        characters, items = await create_characters_and_items(
            result.new_characters, result.new_items
        )
        yield characters
        yield items

        tree_data.environment.add(
            BoringGeneric(
                objects=[{"text": result.consequences}],
                name="story",
            ),
            "action_consequence",
        )

        tree_data.update_tasks_completed(
            prompt=tree_data.user_prompt,
            task="action_consequence",
            num_trees_completed=tree_data.num_trees_completed,
            parsed_info=f"A consequence of the action: {tree_data.user_prompt} has been written.",
        )


class SceneSetter(dspy.Signature):
    """
    You create a completely random, fantasy setting, similar to dungeons and dragons.
    You are a storyteller, an expert one. You create new worlds, new characters, new stories.
    This is the beginning of a dungeons and dragons campaign, where you should be trying to intrigue the player into exploring the world.
    You can set up stories, characters, and settings if you like. Or you can just describe the world in a poetic way.
    You should end the story with a challenge for the player to explore the world.
    Or you can just say something like "what will you do?" (but not exactly these words!).
    Be creative, have fun, be poetic, be stylistic, be engaging.
    """

    user_prompt = dspy.InputField(
        description="""
        The user's prompt. This can be simple, like "start the story", or "create a new character", or "describe the world", or 'let's start'.
        Or it could be a specific description of what the world is like, or what the characters are like, or what the story is about.
        """.strip()
    )

    story: str = dspy.OutputField(
        description="""
        A fantastical introduction to the world and a story.
        """.strip()
    )
    title: str = dspy.OutputField(
        description="""
        A short title for the story.
        """.strip()
    )
    new_characters: list[Character] = dspy.OutputField(
        description="""
        Any new characters that are introduced as a result of what you have written in the `story` field.
        Return a list of characters, this should be populated with _any_ character that you discuss in the `story` field.
        Do not miss a single character, or you will be fired.
        """.strip()
    )
    new_items: list[Item] = dspy.OutputField(
        description="""
        Any new items that are introduced as a result of what you have written in the `story` field.
        Return a list of items, this should be populated with _any_ item that you discuss in the `story` field.
        Do not miss a single item, or you will be fired.
        """.strip()
    )


class SetTheScene(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="set_the_scene",
            description="""
            Set the scene of the story.
            This should ONLY be called once, at the beginning of the interactions with the user.
            OR if the user wants to reset the story (they must SPECIFICALLY say so).
            """,
            end=True,
        )
        self.logger = kwargs.get("logger")

    async def is_tool_available(self, **kwargs):
        tree_data = kwargs.get("tree_data")
        return tree_data.environment.is_empty()

    async def __call__(self, **kwargs):
        descriptor = dspy.ChainOfThought(SceneSetter)
        descriptor = dspy.asyncify(descriptor)
        tree_data = kwargs.get("tree_data")

        yield Status("Setting the scene...")

        result = await descriptor(
            user_prompt=kwargs.get("tree_data").user_prompt,
            lm=kwargs.get("complex_lm"),
        )

        yield Summary(
            text=result.story,
            title=result.title,
        )

        tree_data.environment.add(
            BoringGeneric(
                objects=[{"text": result.story}],
                name="story",
            ),
            "set_the_scene",
        )

        yield Status("Creating characters and items...")
        self.logger.info(f"Creating characters: {result.new_characters}")
        self.logger.info(f"Creating items: {result.new_items}")
        characters, items = await create_characters_and_items(
            result.new_characters, result.new_items
        )
        yield characters
        yield items

        tree_data.update_tasks_completed(
            prompt=tree_data.user_prompt,
            task="set_the_scene",
            num_trees_completed=tree_data.num_trees_completed,
            parsed_info="The story has been created, the characters introduced, the world made.",
        )


# if __name__ == "__main__":

#     from elysia import Tree
#     from elysia.config import settings

#     settings.default_config()

#     tree = Tree(
#         style="Poetic, flamboyant, engaging, you are a dungeon master! Crafting worlds.",
#         agent_description="""
#         You are a dungeon master, crafting a story for the player.
#         Your job is to describe what happens in this fantasy world, and interact with the player.
#         They may choose actions, or they may require you to prompt them for more information.
#         Sometimes, their action may not be descriptive enough, so before choosing any actions, you should ask them to be more specific.
#         Then combine these responses to call on the appropriate tools and use them to create stories.
#         Do not create anything yourself, you just call the correct tools with the correct arguments.
#         """,
#         end_goal="The player/the user's answer has been asked to be more specific, or all the actions outlined in the `user_prompt` have been completed.",
#         branch_initialisation="empty",
#         debug=True,
#     )

#     tree.add_tool(SetTheScene)
#     tree.add_tool(ActionConsequence)
#     tree.add_tool(DiceRoll)

#     tree.run(
#         user_prompt="Hey, let's get started!",
#     )

#     tree("I try to land a kick flip on a sick twig nearby")

#     # tree('"Maeve, why do you think it is me that is the chosen one?"')

#     # tree(
#     #     "I throw the chair at Maeve, I realise she is an impostor, and not the real Maeve. I aim for her neck!"
#     # )
