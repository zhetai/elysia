All payloads that are sent from the decision tree to the frontend have the same structure:
```python
{
	"type": str, 
	"id": str,
	"user_id": str,
	"conversation_id": str, 
	"query_id": str,
	"payload": dict
}
```

Where the `"payload"` dictionary always contains:

```python
{
    "type": str, 
    "metadata": dict
    "objects": list[dict],
    ... # additional fields
}
```
where the additional fields are based on the `"type"` of the output.

## Dynamic Display Types

Elysia has different corresponding `payload_types`, depending on the type of the result that is sent to the frontend. 