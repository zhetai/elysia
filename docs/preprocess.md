# Preprocessing Collections

## Overview

[The `preprocess` function](Reference/Preprocessor.md) must be used on the Weaviate collections you plan to use within Elysia. Preprocessing does several things:

- Creates an LLM generated summary of the collection, including descriptions of the fields in the dataset.
- Creates 'mappings', so that fields in the collection can be mapped to frontend-specific fields. This enables the Elysia frontend app to display items from the collection when retrieved in the app.
- Calculates summary statistics, such as the mean, maximum and minimum values of number fields, as well as statistics for other fields.

## Configuring the Preprocessing Function

Since preprocessing uses LLM created summaries of the collections, you must configure your models in advance. The preprocessor uses the `base_model` for this. The configure can be done via python via

```python
configure(
    base_model="gpt-4o-mini",
    base_provider="openai",
    wcd_url="...",
    wcd_api_key="...",
    logging_level="DEBUG",
)
```
or via environment variables, e.g.
```bash
BASE_MODEL=gpt-4o-mini
BASE_PROVIDER=openai
WCD_URL=...
WCD_API_KEY=...
LOGGING_LEVEL=DEBUG
```

When the logging level is `INFO`, the preprocessor will show a progress bar.

## Running the Preprocessing Function

You have access to two functions, `preprocess_async`, which must be awaited, and `preprocess`, which is a sync wrapper for its async sister. The basic arguments for either function are:

- **`collection_names`** (*list[str])*: The names of the collections to preprocess.
- **`client_manager`** *(ClientManager)*: The client manager to use.
    The ClientManager class is how Elysia interacts with Weaviate client.
    If you are unsure of this, do not provide this argument, it will default to the Weaviate cluster you selected via the `Settings`, or via `configure`/environment variables.

As well, the LLM requires a number of objects retrieved from the collection, at random, to help provide its summary. Since objects in collections vary greatly in token size (and hence LLM compute time/cost), you can adjust the following parameters to change how many objects are used for this sample.

- **`min_sample_size`** *(int)*: The minimum number of objects in the sample.
- **`max_sample_size`** *(int)*: The maximum number of objects to sample.
- **`num_sample_tokens`** *(int)*: The maximum number of tokens in the sample objects used to evaluate the summary.

The `num_sample_tokens` parameter controls how many objects are actually used. Provided it is between `min_sample_size` and `max_sample_size`, the preprocessor will select the closest number of objects that are estimated to be in total `num_sample_tokens` tokens.

Additionally, you have:
- **`settings`** *(Settings)*: The settings to use.
- **`force`** *(bool)*: Whether to force the preprocessor to run even if the collection already exists.

## Additional Functions

You can also use `preprocessed_collection_exists`, which returns True/False if the collection has been preprocessed (and it can be accessed within the Weaviate cluster), and `delete_preprocessed_collection`, which will delete the cached preprocessed metadata.