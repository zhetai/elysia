import datetime
import json
import uuid

from weaviate.collections.classes.aggregate import (
    AggregateBoolean,
    AggregateDate,
    AggregateGroup,
    AggregateGroupByReturn,
    AggregateInteger,
    AggregateNumber,
    AggregateReturn,
    AggregateText,
    GroupByAggregate,
)


def objects_dict_to_str(objects: list) -> str:
    out = ""
    for item in objects:
        if isinstance(item, dict):
            inner = "["
            for key, value in item.items():
                try:
                    inner += json.dumps({key: value}) + ", "
                except TypeError:
                    pass
            inner += "]"
            out += inner + "\n"

        elif isinstance(item, list):
            out += objects_dict_to_str(item)

        elif isinstance(item, str):
            out += item + "\n"

    return out


def format_datetime(dt: datetime.datetime) -> str:
    dt = dt.isoformat("T")
    plus = dt.find("+")
    if plus != -1:
        return dt[:plus] + "Z"
    else:
        return dt + "Z"


def format_dict_to_serialisable(d):
    for key, value in d.items():
        if isinstance(value, datetime.datetime):
            d[key] = format_datetime(value)

        elif isinstance(value, dict):
            format_dict_to_serialisable(value)

        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    format_dict_to_serialisable(item)

                elif isinstance(item, datetime.datetime):
                    d[key][i] = format_datetime(item)

                elif isinstance(item, uuid.UUID):
                    d[key][i] = str(item)

        elif isinstance(value, uuid.UUID):
            d[key] = str(value)


def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


# def update_current_message(current_message: str, new_message: str):
#     """
#     current_message: str - the current message in the conversation history
#     new_message: str - the new message from the LLM
#     The new_message is filtered for the <NEW> tags, and anything enclosed in them is returned as the new sentence.
#     If there are no <NEW> tags, the new message is returned as the new sentence.
#     Returns:
#         [0] - the updated message including the new sentence without any tags
#         [1] - the new sentence
#     """
#     # if this is the first message, remove any tags and return the new message
#     if current_message == "":
#         if "<NEW>" in new_message:
#             new_message = new_message.replace("<NEW>", "")
#         if "</NEW>" in new_message:
#             new_message = new_message.replace("</NEW>", "")

#         return new_message, new_message

#     # otherwise, find new sentence
#     if "<NEW>" in new_message and "</NEW>" in new_message:
#         # check if this is empty
#         new_sentence = new_message.split("<NEW>")[1]
#         new_sentence = new_sentence[: new_sentence.find("</NEW>")]

#         if new_sentence == "":
#             new_sentence = new_message.replace("<NEW>", "").replace("</NEW>", "")
#             return current_message, new_sentence.split(". ")[-1]
#         else:
#             return current_message + " " + new_sentence, new_sentence

#     else:
#         # backup - in case only one tag gets added
#         if "<NEW>" in new_message and "</NEW>" not in new_message:
#             new_sentence = new_message.split("<NEW>")[1]
#             return current_message + " " + new_sentence, new_sentence
#         elif "<NEW>" not in new_message and "</NEW>" in new_message:
#             new_sentence = new_message.split("</NEW>")[0]
#             return current_message + " " + new_sentence, new_sentence.split(". ")[-1]

#         # if no tags are added, return the last sentence of the new message
#         else:
#             return new_message, new_message.split(". ")[-1]


def format_aggregation_property(prop):
    if isinstance(prop, AggregateText):
        out = {"type": "text", "values": []}
        for top_occurence in prop.top_occurrences:
            out["values"].append(
                {
                    "value": top_occurence.count,
                    "field": top_occurence.value,
                    "aggregation": "count",
                }
            )
        return out

    elif isinstance(prop, AggregateNumber):
        out = {"type": "number", "values": []}

        if prop.count is not None:
            out["values"].append(
                {"value": prop.count, "field": None, "aggregation": "count"}
            )

        if prop.maximum is not None:
            out["values"].append(
                {"value": prop.maximum, "field": None, "aggregation": "maximum"}
            )

        if prop.mean is not None:
            out["values"].append(
                {"value": prop.mean, "field": None, "aggregation": "mean"}
            )

        if prop.median is not None:
            out["values"].append(
                {"value": prop.median, "field": None, "aggregation": "median"}
            )

        if prop.minimum is not None:
            out["values"].append(
                {"value": prop.minimum, "field": None, "aggregation": "minimum"}
            )

        if prop.mode is not None:
            out["values"].append(
                {"value": prop.mode, "field": None, "aggregation": "mode"}
            )

        if prop.sum_ is not None:
            out["values"].append(
                {"value": prop.sum_, "field": None, "aggregation": "sum"}
            )

        return out

    elif isinstance(prop, AggregateDate):
        out = {"type": "date", "values": []}

        if prop.count is not None:
            out["values"].append(
                {"value": prop.count, "field": None, "aggregation": "count"}
            )

        if prop.maximum is not None:
            out["values"].append(
                {"value": prop.maximum, "field": None, "aggregation": "maximum"}
            )

        if prop.median is not None:
            out["values"].append(
                {"value": prop.median, "field": None, "aggregation": "median"}
            )

        if prop.minimum is not None:
            out["values"].append(
                {"value": prop.minimum, "field": None, "aggregation": "minimum"}
            )

        if prop.mode is not None:
            out["values"].append(
                {"value": prop.mode, "field": None, "aggregation": "mode"}
            )

        return out
    else:
        return {"type": "unknown", "values": []}


def format_aggregation_response(response):
    out = {}
    if isinstance(response, AggregateGroupByReturn):
        # this gets excluded from the frontend payload
        if "total_count" in dir(response):
            out["ELYSIA_NUM_ITEMS"] = response.total_count

        for result in response.groups:
            field = result.grouped_by.prop
            out[field] = {"type": "text", "values": []}

        for result in response.groups:
            field = result.grouped_by.prop

            if result.total_count is not None:
                out[field]["values"].append(
                    {
                        "value": result.total_count,
                        "field": result.grouped_by.value,
                        "aggregation": "count",
                    }
                )

            for key, prop in result.properties.items():
                formatted_props = format_aggregation_property(prop)
                if "groups" not in out[field]:
                    out[field]["groups"] = {}

                if result.grouped_by.value not in out[field]["groups"]:
                    out[field]["groups"][result.grouped_by.value] = {}

                out[field]["groups"][result.grouped_by.value][key] = formatted_props

    elif isinstance(response, AggregateReturn):
        # this gets excluded from the frontend payload
        if "total_count" in dir(response):
            out["ELYSIA_NUM_ITEMS"] = response.total_count

        for field, field_properties in response.properties.items():
            out[field] = {}
            props = format_aggregation_property(field_properties)
            for key, prop in props.items():
                out[field][key] = prop

    elif isinstance(response, list):
        for item in response:
            out.append(format_aggregation_response(item))

    return out
