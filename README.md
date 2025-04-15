
# Elysia Python Client + API

## Installation (bash) (Linux/MacOS)

From your bash terminal, clone the repository via
```bash
git clone https://github.com/weaviate/elysia
```
move to the working directory
```bash
cd elysia
```
Create a virtual environment with Python (version 3.10 - 3.12, see installation instructions for [brew](https://formulae.brew.sh/formula/python@3.12) or [ubuntu](https://ubuntuhandbook.org/index.php/2023/05/install-python-3-12-ubuntu/)).
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```
and then install Elysia via pip
```bash
pip install -e .
```

### Configuring the Environment

To use Elysia with Weaviate, i.e. for agentic searching and retrieval, you need a Weaviate cluster api key and URL
```
WCD_URL=...
WCD_API_KEY=...
```
Elysia will automatically detect these when running locally, and this will be the default Weaviate cluster for all users logging into the Elysia app. But these can be configured on a user-by-user basis.

Whichever vectoriser you use for your Weaviate collection you will need to specify your corresponding API key, e.g.
```
OPENAI_API_KEY=...
```
These will automatically be added to the headers for the Weaviate client.

Same for whichever model you choose for the LLM in Elysia, so if you are using GPT-4o, for example, specify an `OPENAI_API_KEY`.

The 'default' config for Elysia is to use [OpenRouter](https://openrouter.ai/) to give easy access to a variety of models. So this requires
```
OPENROUTER_API_KEY=...
```
and OpenAI as the vectorisers for the alpha datasets, so you need `OPENAI_API_KEY` too.


## Tools (WIP)

### Displaying Outputs

To display the output from a custom tool to the frontend, a specific display type needs to be specified.
For example, if I created a new tool that shuffled a deck of cards and dealt 5 cards to the user, I would need to define a `ShuffleAndDealCards` tool, such as

```python
class ShuffleAndDealCards(Tool):
    pass # add tool specific methods here
```

The `__call__` method of this tool is what is called when the tool is chosen by the decision agent.
So within the `__call__` method, I will shuffle and deal the cards, e.g.

```python
async def __call__(self, **kwargs):
    deck = shuffle_deck()
    hand = deal_cards(deck)
```

To display this to the frontend, I can choose one of a few pre-built generic display options, e.g.

```python
class ShuffleAndDealCards(Tool):
    def __init__(self):
        pass

    async def __call__(self, tree_data: TreeData):

        tree_data.conversation_history
        hand = get_magic_card(tree_data.user_prompt)

        # deck = shuffle_deck()
        # hand = deal_cards(deck) # hand is a list of dicts with entries {"text": ..., "image_url": ..., ...} which maps with ImageDisplayA
        yield ImageDisplayA(objects=hand)
```

Or I can create a custom class that gives properties that the frontend is aware of, so it knows what to display where.
This is done by overriding the `to_json` method from `Result`.

```python
class CardDisplay(Result):
    def __init__(self, objects: list[dict]):
        super().__init__(objects=objects, type="ImageDisplayA")

    def to_json(objects: list[dict] | None = None):

        if objects is None:
            objects = self.objects

        output_objects = []
        for obj in objects:
            output_objects.append(
                {
                    "text": obj["card_text"],
                    "image_url": obj["card_art_image"],
                    "upper_right": obj["card_value"],
                    "lower_left": obj["card_value"],
                }
            )

        return output_objects
```

Or more simply:

```python
yield ImageDisplayA(
    objects = hand,
    mapping = {
        "card_text": "text",
        "card_art_image": "image_url",
        "card_value": ["upper_right", "lower_left"]
    }
)
```

### Parsing Outputs in the LLM

Of course, displaying to the frontend is only half of the specification you need in Elysia.
The decision agent at the decision tree also needs to be aware of what outputs have been returned.
Any outputs yielded by a tool under the superclass `Result` gets its objects automatically added to the `environment` variable that the LLM reads.
This includes outputs like `ImageDisplayA`.
But there is another variable used to help the LLM parse information about retrieved tasks, within the `llm_message` argument of initialising the `Result`.
This has a few existing placeholders:
    - `{type}`: The type of the object
    - `{num_objects}`: The number of objects in the object

As well as any items in the `metadata` dictionary which can be populated also on init.
    - `{metadata_key}`: Any key in the metadata dictionary

```python
yield ImageDisplayA(
    objects = hand,
    mapping = {
        "card_text": "text",
        "card_art_image": "image_url",
        "card_value": ["upper_right", "lower_left"]
    },
    metadata = {
        "low": 1,
        "high": 52
    },
    llm_message = "Dealt {num_objects}, possible range: {low} to {high}"
)
```

This final `ImageDisplayA`, eventually will have two outputs to the frontend and the LLM respectively:
```python
# FRONTEND OUTPUT:
{
    'type': 'result',
    'user_id': 'test',
    'conversation_id': 'test',
    'query_id': 'test',
    'id': 'Ima-dfef396c-decd-45e6-b57e-0e2328417e00',
    'payload': {
        'type': 'ImageDisplayA',
        'objects': [
            {
                'text': 'This is the 28th card',
                'upper_right': 28, 'lower_right': 28
            },
            {
                'text': 'This is the 50th card',
                'upper_right': 50,
                'lower_right': 50
            },
            {
                'text': 'This is the 38th card',
                'upper_right': 38,
                'lower_right': 38
            },
            {
                'text': 'This is the 19th card',
                'upper_right': 19,
                'lower_right': 19
            },
            {
                'text': 'This is the 31th card',
                'upper_right': 31,
                'lower_right': 31
            }
        ],
        'metadata': {
            'low': 1,
            'high': 52
        }
    }
}


# LLM PARSED OUTPUT:
"Dealt 5 cards, possible range: 1 to 52"
```
The LLM can see both the 'objects' (within the environment) and the parsed output.
The parsed output is what is shown underneath a `tasks_completed` input to the LLM, which is broken down into stages.
I.e. the parsed output will appear under its corresponding prompt and task number that it was completed at.
