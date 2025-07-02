# Welcome to Elysia

Welcome to Elysia, the agentic platform for searching and retrieving data in Weaviate. Elysia is also designed to handle any custom tools, and it will be automatically handled by a decision agent.

See the [basic example to get started right away!](basic.md) Or if you want to make your own tools and customise Elysia, see the [Tool Construction Overview](Customising/creating_your_own_tools.md).

## Get Started

Elysia can be used very simply:

```python
import elysia
tree = elysia.Tree()
response, objects = tree(
    "What are the 10 most expensive items in the Ecommerce collection?"
)
```

Note that to use your Weaviate collections with these built in tools, you will need to preprocess them. You also need to configure your models and API keys.

[See the setup page to get started.](setting_up.md)

## Installation

```bash
pip install elysia
```

## Usage

Elysia is free, open source, and available to anyone.

Unlike other agent-based packages, Elysia is pre-configured to run a wide range of tools and has a lot of capabilities straight away. For example, you could just call Elysia on your Weaviate collections and it will immediately and dynamically search your data, using custom queries with filters or aggregations.

Or you could customise Elysia to your liking, create your own custom tools and add them to the Elysia decision tree.


To use Elysia to search your data, you need a Weaviate cluster (or you can define your own custom tool to search another data source!).

[Sign up to Weaviate! A 14 day sandbox cluster is free.](https://weaviate.io/deployment/serverless)

For more information on signing up to Weaviate, [click here](https://weaviate.io/developers/wcs/platform/create-account). 

From your weaviate cluster, you can upload data via a CSV on the cloud console, or [you can upload via the Weaviate APIs](https://weaviate.io/developers/academy/py/zero_to_mvp/schema_and_imports/import).

## About

Check out the Github Repositories for the backend and the frontend

- [elysia](https://github.com/weaviate/elysia) (backend)

- [elysia-frontend](https://github.com/weaviate/elysia-frontend) (frontend)

Elysia was developed by Edward Schmuhl (frontend) and Danny Williams (backend). Check out our socials below:


- [Edward's Linkedin](https://www.linkedin.com/in/edwardschmuhl/)

- [Danny's Linkedin](https://www.linkedin.com/in/dannyjameswilliams/)

Documentation built with [mkdocs](https://www.mkdocs.org).