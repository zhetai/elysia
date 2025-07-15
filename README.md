# Welcome to Elysia

Elysia is an agentic platform designed to use tools in a decision tree. A decision agent decides which tools to use dynamically based on its environment and context. You can use custom tools or use the pre-built tools designed to retrieve your data in a Weaviate cluster.

[Read the docs!](https://weaviate.github.io/elysia/)

## Get Started

To use Elysia, you need to either set up your models and API keys in your `.env` file, or specify them in the config. [See the setup page to get started.](https://weaviate.github.io/elysia/setting_up/)

Elysia can be used very simply:
```python
from elysia import tool, Tree

tree = Tree()

@tool(tree=tree)
async def add(x: int, y: int) -> int:
    return x + y

tree("What is the sum of 9009 and 6006?")
```

Elysia is pre-configured to be capable of connecting to and interacting with your [Weaviate](https://weaviate.io/deployment/serverless) clusters!
```python
import elysia
tree = elysia.Tree()
response, objects = tree(
    "What are the 10 most expensive items in the Ecommerce collection?",
    collection_names = ["Ecommerce"]
)
```
This will use the built-in open source _query_ tool or _aggregate_ tool to interact with your Weaviate collections. To get started connecting to Weaviate, [see the setting up page in the docs](https://weaviate.github.io/elysia/setting_up/).

## Installation (bash) (Linux/MacOS)

### PyPi (Recommended)
Simply run 
```bash
pip install elysia-ai
```
to install straight away!

### GitHub

To get the latest development version, you can do
```bash
git clone https://github.com/weaviate/elysia
```
move to the working directory via
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
Done! You can now use the Elysia python package


## Running the Elysia App

### Backend 

Just simply run
```bash
elysia start
```
and the backend will start on [localhost:3000](localhost:3000) by default.

### Frontend Installation

From the directory you just cloned, (i.e. inside `elysia/`). You need to clone the [frontend repository](https://github.com/weaviate/elysia-frontend) (give it a star!) via
```bash
git clone https://github.com/weaviate/elysia-frontend
```
navigate inside the frontend repository via
```bash
cd elysia-frontend
```
and install and run the node server via
```bash
npm install
npm run dev
```
By default this will run the app on [localhost:8000](localhost:8000). Visit this to use the app!

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