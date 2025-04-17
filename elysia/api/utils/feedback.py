from datetime import datetime, timedelta

from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Filter, Metrics
from weaviate.util import generate_uuid5

from elysia.objects import Error
from elysia.tree.tree import Tree
from elysia.util.parsing import format_datetime
from elysia.api.core.logging import logger
import weaviate.classes as wvc
import weaviate.classes.config as wc


def create_feedback_collection(client):
    client.collections.create(
        "ELYSIA_FEEDBACK__",
        properties=[
            # session data
            wc.Property(
                name="user_id",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="conversation_id",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="query_id",
                data_type=wc.DataType.TEXT,
            ),
            # feedback value (between -2, -1, 1, 2)
            wc.Property(
                name="feedback",
                data_type=wc.DataType.NUMBER,
            ),
            # Tree data (except available_information)
            wc.Property(
                name="user_prompt",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="conversation_history",
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="role",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="content",
                        data_type=wc.DataType.TEXT,
                    ),
                ],
            ),
            wc.Property(
                name="tasks_completed",
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="prompt",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="tasks",
                        data_type=wc.DataType.OBJECT_ARRAY,
                        nested_properties=[
                            wc.Property(
                                name="task",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="reasoning",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="todo",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(name="action", data_type=wc.DataType.BOOL),
                            wc.Property(
                                name="count",
                                data_type=wc.DataType.NUMBER,
                            ),
                            wc.Property(
                                name="extra_string",
                                data_type=wc.DataType.TEXT,
                            ),
                        ],
                    ),
                ],
            ),
            wc.Property(
                name="current_message",
                data_type=wc.DataType.TEXT,
            ),
            # Extra specifics
            wc.Property(
                name="route",
                data_type=wc.DataType.TEXT_ARRAY,
            ),
            wc.Property(
                name="action_information",
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="collection_name",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="action_name",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="code",
                        data_type=wc.DataType.OBJECT,
                        nested_properties=[
                            wc.Property(
                                name="title",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="language",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="text",
                                data_type=wc.DataType.TEXT,
                            ),
                        ],
                    ),
                    wc.Property(
                        name="return_type",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="output_type",
                        data_type=wc.DataType.TEXT,
                    ),
                ],
            ),
            # metadata
            wc.Property(
                name="time_taken_seconds",
                data_type=wc.DataType.NUMBER,
            ),
            wc.Property(
                name="base_lm_used",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="complex_lm_used",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="feedback_date",
                data_type=wc.DataType.DATE,
            ),
            # proper things for training models
            wc.Property(name="training_updates", data_type=wc.DataType.TEXT),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
    )
    logger.info("Feedback collection (ELYSIA_FEEDBACK__) created!")


async def create_feedback(
    user_id: str, conversation_id: str, query_id: str, feedback: int, tree: Tree, client
):
    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

    history = tree.history[query_id]

    # new_tasks_completed = []
    # for task_prompt in history["tree_data"].tasks_completed:
    #     new_tasks_completed.append({})
    #     new_tasks_completed[-1]["prompt"] = task_prompt["prompt"]
    #     new_tasks_completed[-1]["task"] = [
    #         {
    #             "task": h["task"],
    #             "reasoning": h["reasoning"],
    #             "todo": h["todo"],
    #             "count": float(h["count"])
    #         }
    #     for h in task_prompt["tasks"]
    #     ]

    # ensure "action" is a bool in tasks_completed
    for task_prompt in history["tree_data"].tasks_completed:
        for task in task_prompt["task"]:
            task["action"] = bool(task["action"])
            del task["inputs"]
            # TODO: remake feedback collection later to include inputs

    date_now = datetime.now()
    date_now = date_now - timedelta(days=1)

    properties = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query_id": query_id,
        "feedback": int(feedback),
        "user_prompt": history["tree_data"].user_prompt,
        "conversation_history": history["tree_data"].conversation_history,
        "tasks_completed": history["tree_data"].tasks_completed,
        "current_message": history["tree_data"].current_message,
        "route": history["decision_history"],
        "action_information": history["action_information"],
        "time_taken_seconds": history["time_taken_seconds"],
        "base_lm_avg_time": tree.base_lm_timer.avg_time,
        "complex_lm_avg_time": tree.complex_lm_timer.avg_time,
        "base_lm_used": tree.base_lm.model if tree.debug else tree.base_lm,
        "complex_lm_used": tree.complex_lm.model if tree.debug else tree.complex_lm,
        "feedback_datetime": format_datetime(date_now),
        "feedback_date": format_datetime(date_now),
        "training_updates": history["training_updates"],
        "initialisation": history["initialisation"],
    }

    # from rich import print
    # print(history["training_updates"])

    # uuid is generated based on the user_id, conversation_id, query_id ONLY
    # so if the user re-selects the same feedback, it will be updated instead of added
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )

    try:
        if await feedback_collection.data.exists(session_uuid):
            await feedback_collection.data.update(
                properties=properties, uuid=session_uuid
            )
        else:
            await feedback_collection.data.insert(
                properties=properties, uuid=session_uuid
            )
    except Exception as e:
        logger.exception(f"Whilst inserting feedback to the feedback collection")


async def view_feedback(user_id: str, conversation_id: str, query_id: str, client):
    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
        return Error("No feedback collection found")

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )

    feedback_object = await feedback_collection.query.fetch_object_by_id(
        uuid=session_uuid
    )

    return feedback_object.properties


async def remove_feedback(user_id: str, conversation_id: str, query_id: str, client):
    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
        return Error("No feedback collection found")

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )
    await feedback_collection.data.delete_by_id(uuid=session_uuid)


async def feedback_metadata(client, user_id: str):
    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

    if user_id != "admin":
        user_id_filter = Filter.by_property("user_id").equal(user_id)
    else:
        user_id_filter = None

    all_aggregate = await feedback_collection.aggregate.over_all(
        filters=user_id_filter, total_count=True
    )
    total_feedback = all_aggregate.total_count

    # get unique feedback values
    # feedback_values = await feedback_collection.aggregate.over_all(
    #     group_by = GroupByAggregate(prop="feedback"),
    #     filters = user_id_filter,
    #     return_metrics = [
    #         Metrics("feedback").number(
    #             count=True
    #         )
    #     ]
    # )

    feedback_values = {
        0.0: "negative",
        1.0: "positive",
        2.0: "superpositive",
    }  # [float(group.grouped_by.value) for group in feedback_values.groups]

    feedback_by_date = {}
    feedback_by_value = {}
    for feedback_value in [0.0, 1.0, 2.0]:
        if user_id != "admin":
            filters = Filter.all_of(
                [Filter.by_property("feedback").equal(feedback_value), user_id_filter]
            )
        else:
            filters = Filter.by_property("feedback").equal(feedback_value)

        agg_feedback_i = await feedback_collection.aggregate.over_all(
            group_by=GroupByAggregate(prop="feedback_date"),
            filters=filters,
            return_metrics=[Metrics("feedback").number(count=True)],
        )

        agg_feedback_count_i = await feedback_collection.aggregate.over_all(
            filters=filters, return_metrics=[Metrics("feedback").integer(count=True)]
        )
        feedback_by_value[feedback_values[feedback_value]] = (
            agg_feedback_count_i.properties["feedback"].count
        )

        for date_group in agg_feedback_i.groups:
            if date_group.grouped_by.value not in feedback_by_date:
                feedback_by_date[date_group.grouped_by.value] = {}
            feedback_by_date[date_group.grouped_by.value][
                feedback_values[feedback_value]
            ] = date_group.properties["feedback"].count

    for date in feedback_by_date:
        for feedback_name in feedback_values.values():
            if feedback_name not in feedback_by_date[date]:
                feedback_by_date[date][feedback_name] = 0

    agg_feedback = await feedback_collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="feedback_date"),
        filters=user_id_filter,
        return_metrics=[Metrics("feedback").number(mean=True, count=True)],
    )

    for date_group in agg_feedback.groups:
        if date_group.grouped_by.value not in feedback_by_date:
            feedback_by_date[date_group.grouped_by.value] = {}
        feedback_by_date[date_group.grouped_by.value]["mean"] = date_group.properties[
            "feedback"
        ].mean
        feedback_by_date[date_group.grouped_by.value]["count"] = date_group.properties[
            "feedback"
        ].count

    if user_id != "admin":
        filters = Filter.all_of(
            [Filter.by_property("base_lm_avg_time").greater_than(0.0), user_id_filter]
        )
    else:
        filters = Filter.by_property("base_lm_avg_time").greater_than(0.0)

    agg_query_speed_by_base_model = await feedback_collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="base_lm_used"),
        filters=filters,
        return_metrics=[
            Metrics("base_lm_avg_time").number(mean=True, maximum=True, minimum=True)
        ],
    )

    query_speed_by_base_model = {}
    for model_group in agg_query_speed_by_base_model.groups:
        query_speed_by_base_model[model_group.grouped_by.value] = {
            "mean": model_group.properties["base_lm_avg_time"].mean,
            "maximum": model_group.properties["base_lm_avg_time"].maximum,
            "minimum": model_group.properties["base_lm_avg_time"].minimum,
        }

    if user_id != "admin":
        filters = Filter.all_of(
            [
                Filter.by_property("complex_lm_avg_time").greater_than(0.0),
                user_id_filter,
            ]
        )
    else:
        filters = Filter.by_property("complex_lm_avg_time").greater_than(0.0)

    agg_query_speed_by_complex_model = await feedback_collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="complex_lm_used"),
        filters=filters,
        return_metrics=[
            Metrics("complex_lm_avg_time").number(mean=True, maximum=True, minimum=True)
        ],
    )

    query_speed_by_complex_model = {}
    for model_group in agg_query_speed_by_complex_model.groups:
        query_speed_by_complex_model[model_group.grouped_by.value] = {
            "mean": model_group.properties["complex_lm_avg_time"].mean,
            "maximum": model_group.properties["complex_lm_avg_time"].maximum,
            "minimum": model_group.properties["complex_lm_avg_time"].minimum,
        }

    agg_full_query_time = await feedback_collection.aggregate.over_all(
        filters=user_id_filter,
        return_metrics=[
            Metrics("time_taken_seconds").number(mean=True, maximum=True, minimum=True)
        ],
    )

    full_query_time = {
        "mean": agg_full_query_time.properties["time_taken_seconds"].mean,
        "maximum": agg_full_query_time.properties["time_taken_seconds"].maximum,
        "minimum": agg_full_query_time.properties["time_taken_seconds"].minimum,
    }

    return {
        "total_feedback": total_feedback,
        "feedback_by_value": feedback_by_value,
        "feedback_by_date": feedback_by_date,
        "call_speed_by_base_model": query_speed_by_base_model,
        "call_speed_by_complex_model": query_speed_by_complex_model,
        "full_query_time": full_query_time,
    }


# if __name__ == "__main__":
#     import os
#     import weaviate
#     from weaviate.classes.init import Auth
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url = os.environ.get("WCD_URL"),
#         auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
#         headers={"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
#     )
#     print(feedback_metadata(client))
