from typing import List

import dspy

from elysia.tree.objects import Atlas
from elysia.tools.retrieval.util import (
    AggregationOutput,
    CollectionSchema,
    DataDisplay,
    QueryOutput,
    ListQueryOutputs,
    ListAggregationOutputs,
)
from typing import Union


class AggregationPrompt(dspy.Signature):
    """
    You are a database query builder specializing in converting natural language questions into structured aggregation queries for Weaviate databases.

    You are part of an ensemble, a larger system of an agentic framework the user is interacting with, so you are not the only LLM they interact with.

    Given a user question, your job is to formulate precise aggregation queries based on the provided schema information.

    The AggregationOutput format contains these key components:

    1. target_collections: The collections you want to query (can be multiple, all will share the same aggregation query, but different queries can be chosen by providing multiple AggregationOutput objects)
    2. search_query: Optional text to search for when filtering data for aggregation. This will filter data before aggregation via semantic search, so use sparingly.

    3. Filtering mechanism:
    - filter_buckets: Groups of filters connected with AND/OR logic
    - Each filter_bucket contains:
        - filters: List of property-specific filters or nested filter buckets
        - operator: "AND" or "OR" to connect filters within the bucket

    The FilterBucket is recursive - it can contain other FilterBuckets. This allows you to build complex logical expressions like (A AND B) OR (C AND D) by nesting buckets with different operators. For example:
    - A top-level bucket with OR operator containing:
        - A nested bucket with AND operator (for A AND B)
        - Another nested bucket with AND operator (for C AND D)
    - Example: For "January OR March", create a structure like: OR(AND(Jan-start, Jan-end), AND(Mar-start, Mar-end))

    4. Property filter types:
        - IntPropertyFilter: For numeric comparisons (=, <, >, <=, >=)
        - TextPropertyFilter: For text equality (=) or pattern matching (LIKE)
        - BooleanPropertyFilter: For boolean comparisons (=, !=)
        - DatePropertyFilter: For date comparisons (=, <, >, <=, >=)
       Make sure to use the correct filter type for the property type. The property type is given in the schema (field.[field_name].type).
       If you use the incorrect filter type, the query will error. It is extremely important that you use the correct filter type!

    5. Aggregation operations (can choose many per AggregationOutput object, match the property type based on the field in the schema):
        - integer_property_aggregation: MIN, MAX, MEAN, MEDIAN, MODE, SUM on numeric fields
        - text_property_aggregation: TOP_OCCURRENCES for most frequent values
        - boolean_property_aggregation: TOTAL_TRUE, TOTAL_FALSE, PERCENTAGE_TRUE, PERCENTAGE_FALSE
        - date_property_aggregation: MIN, MAX, MEAN, MEDIAN, MODE on date fields
        Make sure to use the correct aggregation operation for the property type. The property type is given in the schema (field.[field_name].type).
        If you use the incorrect aggregation operation, the query will error. It is extremely important that you use the correct aggregation operation!

    6. Additional parameters:
    - limit: Maximum number of results to return (default: 5), only relevant when using search_query
    - groupby_property: Group aggregation results by a specific property value
        Grouping is optional but recommended for complex questions or those that require segmentation of results

    When constructing aggregation queries:
    1. Identify the relevant collection(s) from the schemas provided
    2. Choose the appropriate aggregation operation based on the user's intent (MIN, MAX, TOP_OCCURRENCES, etc.)
    3. Build filter_buckets when specific conditions are needed, using nested buckets for complex logic
    4. Apply grouping when needed to segment the aggregation results

    Provide detailed reasoning about your approach, explaining why you chose specific collections, properties, operations, and filters.
    In your reasoning also be detailed about the nested conditions of filter buckets.
    Include all of your thoughts in any reasoning fields, as this will be used to communicate to later models about your decision process here.

    Return a list of AggregationOutput objects that satisfy the user's request. If multiple approaches are possible, return multiple queries. If the aggregation is impossible to execute, set impossible=True and return None.

    Remember that this is NOT the only query being performed in the larger program! So if you need to aggregate as a first or second step towards achieving a goal,
    you should do so, and explain in your reasoning what your future steps are towards this goal.
    You can only aggregate, which is creating summary statistics of data.
    If you think the goal is better suited to another task such as searching, you should still create an aggregation query to answer the question as best as you can.

    IMPORTANT NOTE: any query you perform will be on the FULL collection, not a filtered subset from any previous steps.
    Do not enclose any answers in ```python or ```.
    """

    available_collections: list[str] = dspy.InputField(
        description="A list of collections that are available to query. You can ONLY choose from these collections."
    )
    previous_aggregation_queries: list[str] = dspy.InputField(
        desc="""
        For each collection, a list of existing code that has been used to query the collection.
        This can be used to avoid generating duplicate queries.
        If any of the collections have an empty list, you are generating the first query for that collection.
        """.strip(),
        format=list[dict],
    )
    aggregation_queries: Union[ListAggregationOutputs, None] = dspy.OutputField(
        description="The aggregation query(-ies) to be executed. Return None if aggregation is impossible to execute"
    )


class QueryCreatorPrompt(dspy.Signature):
    """
    You are a database query builder specializing in converting natural language questions into structured queries.

    You are part of an ensemble, a larger system of an agentic framework the user is interacting with, so you are not the only LLM they interact with.

    Given a user question, your job is to formulate precise queries based on the provided schema information.

    The QueryOutput format contains these key components:

    1. target_collection: The collection you want to query
    2. search_type: Choose from:
       - "hybrid": Combined keyword and vector search
       - "keyword": Pure text-matching search
       - "vector": Semantic search using embeddings
       - "filter_only": Standard SQL style search with filters, sorting, limits, fetching etc.

    3. search_query: The text to search for (optional, not used with filter_only)

    4. Filtering mechanism:
       - filter_buckets: Groups of filters connected with AND/OR logic
       - Each filter_bucket contains:
         - filter: List of property-specific filters or nested filter buckets
         - operator: "AND" or "OR" to connect filters within the bucket

       The FilterBucket is recursive - it can contain other FilterBuckets. This allows you to build complex logical expressions like (A AND B) OR (C AND D) by nesting buckets with different operators. For example:
       - A top-level bucket with OR operator containing:
         - A nested bucket with AND operator (for A AND B)
         - Another nested bucket with AND operator (for C AND D)
       - Example: For "January OR March", create a structure like: OR(AND(Jan-start, Jan-end), AND(Mar-start, Mar-end))

    5. Property filter types:
       - IntPropertyFilter: For numeric comparisons (=, <, >, <=, >=)
       - TextPropertyFilter: For text equality (=) or pattern matching (LIKE)
       - BooleanPropertyFilter: For boolean comparisons (=, !=)
       - DatePropertyFilter: For date comparisons (=, <, >, <=, >=)
       Make sure to use the correct filter type for the property type. The property type is given in the schema (field.[field_name].type).
       If you use the incorrect filter type, the query will error. It is extremely important that you use the correct filter type!

    6. Additional query parameters:
       - sort_by: Property and direction for sorting results
       - limit: Maximum number of results to return (default: 5)
       - groupby_property: Group results by a specific property value

    When constructing queries:
    1. Identify the relevant collection from the schemas provided
    2. Choose the appropriate search_type based on the user's intent
    3. Build filter_buckets when specific conditions are needed, using nested buckets for logic
    4. Apply sorting and limits as needed

    Tips for query_output:
    1.  a. Use 'hybrid' if you are unsure of the search type (but still need a search term), as this combines semantic and keyword search.
        b. Use 'filter_only' if there are no search terms needed.
        c. 'vector' and 'keyword' are good but should be reserved for specific cases where they are needed.
            e.g. keyword is good for looking for when the search terms WILL appear in the text, but do NOT use this as a filter if you can use a field instead
                 vector is good for matching meaning only, and the words themselves are not that important

    2. Err on the side of using larger limits, because you can parse the results later, and it's better to have more than too few.
    3. Do not use a search term to look for specific categories of items if there is a corresponding field in the schema you can use instead as a filter.
    4. You cannot combine sorting with a search term (mutually exclusive), so if sorting is important, do not specify a search term.
    5. Nesting filter buckets is really powerful, and if the queries are particularly complex, definitely use this feature.
    6. Do not be afraid of using lots of features (filters, limits, search terms) at once, as this may be required for bigger queries.
    7. Do not feel the need to answer everything all at once. You are just one step in a larger program. This step can be repeated at a later date.
    8. If you are querying multiple collections for the same thing, do not repeat the same query for each collection, just provide multiple collections a single query_output.
    9. Always make sure your filter matches the property type in the dataset.

    Include all of your thoughts in any reasoning fields, as this will be used to communicate to later models about your decision process here.

    Return a list of QueryOutput objects that satisfy the user's request. If multiple approaches are possible, return multiple queries.

    Remember that this is NOT the only query being performed in the larger program! So if you need to query as a first or second step towards achieving a goal,
    you should do so, and explain in your reasoning what your future steps are towards this goal.

    IMPORTANT NOTE: any query you perform will be on the FULL collection, not a filtered subset from any previous steps.
    So ensure that you use consistent filters and other arguments across all queries.
    Do not enclose any answers in ```python or ```.
    """

    # In your reasoning, give full debug output. Explain every single step in excruciating detail.
    # If you have a destination ID, explain where you obtained it from, in detail. Which field, where, etc.
    available_collections: list[str] = dspy.InputField(
        description="A list of collections that are available to query. You can ONLY choose from these collections."
    )
    previous_queries: list[str] = dspy.InputField(
        desc="""
        For each collection, a list of existing code that has been used to query the collection.
        This can be used to avoid generating duplicate queries.
        If any of the collections have an empty list, you are generating the first query for that collection.
        """.strip(),
        format=list[dict],
    )
    collection_display_types: dict = dspy.InputField(
        desc="""
        A dictionary of the display types for each collection. Use this to determine the display type in `display_type`, for each collection.
        The output of `display_type` must be one of the values in the list for the dictionary entry for that collection.
        """.strip(),
    )

    # Output fields
    query_output: Union[ListQueryOutputs, None] = dspy.OutputField(
        description="The query(-ies) to be executed. Return None if query is impossible to execute"
    )
    data_display: dict[str, DataDisplay] = dspy.OutputField(
        desc="""
        Some metadata about the data you will be returning.
        The keys are the collection names, and the values are the DataDisplay objects.
        The `display_type` should match ONE of the display types in the `collection_display_types` for the corresponding collection.
        The `summarise_items` should be True if the user has specifically asked for summaries of the objects (in an itemised fashion, not an overall summary).
        Otherwise, it should be False.
        """.strip()
    )


class ObjectSummaryPrompt(dspy.Signature):
    """
    Given a list of objects (a list of dictionaries, where each item in the dictionary is a field from the object),
    you must provide a list of strings, where each string is a summary of the object.

    These objects can be of any type, and you should summarise them in a way that is useful to the user.
    """

    objects: list[dict] = dspy.InputField(
        desc="The objects to summarise.", format=list[dict]
    )
    summaries: list[str] = dspy.OutputField(
        desc="""
        The summaries of each individual object, in a list of strings.
        These summaries should be extremely concise, and not aiming to summarise all objects, but a brief summary/description of _each_ object.
        Specifically, what the object is, what it is about to give the user a concise idea of the individual objects. A few sentences at most.
        Your output should be a list of strings in Python format, e.g. `["summary_1", "summary_2", ...]`.
        Do not enclose with ```python or ```. Just the list only.
        """.strip(),
        format=list[str],
    )
