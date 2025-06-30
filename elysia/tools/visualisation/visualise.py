from logging import Logger
from elysia.objects import Tool
from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
from elysia.tools.visualisation.objects import (
    ChartResult,
    BarChart,
    HistogramChart,
    ScatterOrLineChart,
)
from elysia.tools.visualisation.util import convert_chart_types_to_matplotlib
from elysia.tools.visualisation.prompt_templates import (
    CreateBarChart,
    CreateHistogramChart,
    CreateScatterChart,
    CreateLineChart,
)
from elysia.objects import Response, Reasoning
from elysia.util.objects import TrainingUpdate, TreeUpdate


class Visualise(Tool):

    def __init__(self, logger: Logger | None = None, **kwargs):
        super().__init__(
            name="visualise",
            description=(
                "Visualise data in a chart from the environment. "
                "You can only visualise data that is in the environment. "
                "If there is nothing relevant in the environment, do not choose this tool."
            ),
            status="Visualising...",
            end=True,
            inputs={
                "chart_type": {
                    "type": "string",
                    "description": (
                        "The type of chart to create. "
                        "Must be one of: bar, histogram, scatter, line."
                    ),
                    "required": True,
                    "default": "",
                }
            },
        )
        self.logger = logger

    async def __call__(
        self,
        tree_data,
        inputs: dict,
        base_lm,
        complex_lm,
        client_manager,
        **kwargs,
    ):
        chart_type = inputs["chart_type"]

        type_mapping = {
            "bar": CreateBarChart,
            "histogram": CreateHistogramChart,
            "scatter": CreateScatterChart,
            "line": CreateLineChart,
        }

        create_chart = ElysiaChainOfThought(
            type_mapping[chart_type],
            tree_data=tree_data,
            environment=True,
            impossible=True,
            tasks_completed=True,
        )
        prediction = await create_chart.aforward(lm=base_lm)
        charts = prediction.charts
        title = prediction.overall_title

        yield Response(text=prediction.message_update)
        yield Reasoning(reasoning=prediction.reasoning)
        yield TrainingUpdate(
            module_name="visualise",
            inputs={
                "chart_type": chart_type,
                **tree_data.to_json(),
            },
            outputs=prediction.__dict__["_store"],
        )

        if prediction.impossible:
            yield ChartResult(
                [],
                chart_type,
                title,
                metadata={
                    "impossible": True,
                    "impossible_reasoning": prediction.reasoning,
                },
            )
            return

        if self.logger and self.logger.level <= 20:
            fig = convert_chart_types_to_matplotlib(charts, chart_type)
            fig.show()

        yield ChartResult(charts, chart_type, title)
