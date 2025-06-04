from elysia import Tool
from elysia.objects import Response


class TellAJoke(Tool):
    """
    Example tool for testing/demonstration purposes.
    Simply returns a joke as a text response that was an input to the tool.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="tell_a_joke",
            description="Displays a joke to the user.",
            inputs={
                "joke": {
                    "type": str,
                    "description": "A joke to tell.",
                    "required": True,
                }
            },
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        yield Response(inputs["joke"])
