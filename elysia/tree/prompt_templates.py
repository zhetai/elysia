from typing import Literal

import dspy


def construct_decision_prompt(
    available_tasks: list[dict] | None = None,
) -> dspy.Signature:
    ActionLiteral = (
        Literal[tuple(available_tasks)]
        if (available_tasks is not None and len(available_tasks) > 0)  # type: ignore
        else str
    )

    class DecisionPrompt(dspy.Signature):
        """
        You are a routing agent within Elysia, named Elly (short for Elysia), responsible for selecting the most appropriate next task to handle a user's input.
        Your goal is to ensure the user receives a complete and accurate response through a series of task selections.
        You also respond to the user.

        Core Decision Process:
        1. Analyze the user's input prompt and available tasks
        2. Review completed tasks and their outcomes in previous_reasoning
        3. Check if current information satisfies the input prompt
        4. Select the most appropriate next task from available_tasks
        5. Determine if all possible actions have been exhausted

        Decision Rules:
        - Always select from available_tasks list only
        - Prefer tasks that directly progress toward answering the input prompt
        - Consider tree_count to avoid repetitive decisions
        """

        # Regular input fields
        user_prompt: str = dspy.InputField(
            description="The user's original question/prompt that needs to be answered"
        )
        instruction: str = dspy.InputField(
            description="Specific guidance for this decision point that must be followed"
        )
        reference: dict = dspy.InputField(
            description="Current context information (e.g., date, time) for decision-making"
        )
        conversation_history: list[dict] = dspy.InputField(
            description="""
            Previous messages between user and assistant in chronological order:
            [{"role": "user"|"assistant", "content": str}]
            Use this to maintain conversation context and avoid repetition.
            """.strip()
        )

        # Collection information for user to ask basic questions about collection
        collection_information: dict = dspy.InputField(
            description="""
            Metadata about available collections:
            { name: { summary, fields: { min, max, type, etc. } }
            Where name is the name of the collection, and summary is a short description of the collection.
            The statistics such as min/max is of the data in the field. If this is numerical, this is the range of values in the data.
            If string, it is the length of the string. If it is a list, it is related to the length of the list.
            Use to determine if user's request is possible with available data.
            """.strip()
        )

        tree_count: str = dspy.InputField(
            description="""
            Current attempt number as "X/Y" where:
            - X = current attempt number
            - Y = maximum allowed attempts
            Consider ending the process as X approaches Y.
            """.strip()
        )

        tasks_completed: str = dspy.InputField(
            description="""
            Which tasks have been completed in order. These are numbered so that higher numbers are more recent.
            Separated by prompts (so you should identify the prompt you are currently working on to see what tasks have been completed so far)
            Also includes reasoning for each task, to continue a decision logic across tasks.
            Use this to determine whether future searches, for this prompt are necessary, and what task(s) to choose.
            It is IMPORTANT that you separate what actions have been completed for which prompt, so you do not think you have failed an attempt for a different prompt.
            """.strip()
        )

        # Task-specific input fields
        available_actions: list[dict] = dspy.InputField(
            description="""
            List of possible actions to choose from:
            {
                "[name]": {
                    "function_name": [name of the function],
                    "description": [task description],
                    "future": [future information],
                    "inputs": [inputs for the action]. A dictionary where the keys are the input names, and the values are the input descriptions.
                }
            }
            """.strip()
        )

        environment: str = dspy.InputField(
            description="""
            Information gathered from completed tasks.
            Empty if no data has been retrieved yet.
            Use to determine if more information is needed.
            All items here are already shown to the user, so do not repeat information from these fields unless summarising, providing extra information or otherwise.
            E.g., do not list out anything from here, only provide new content to the user.
            """.strip(),
            format=str,
        )

        # Output fields
        # evaluation: str = dspy.OutputField(
        #     description="""
        #     Write out your evaluation of the current state of the decision process.
        #     Re-write what tasks have been completed so far, and what they have accomplished. If you are uncertain, state that.
        #     Evaluate what tasks have been repeated, to avoid getting stuck in a loop.
        #     Evaluate what the last task that was completed was, and what the next task should be.
        #     For the next task, write out each task possible and whether each should be picked and why.
        #     """.strip()
        # )

        function_name: ActionLiteral = dspy.OutputField(
            description="""
            Select exactly one function name from available_actions that best advances toward answering the user's input prompt.
            You MUST select one function name exactly as written as it appears in the `function_name` field of the dictionary.
            You can select a function again in the future if you need to, you are allowed to choose the same function multiple times.
            If unsure, you should try to complete an action that is related to the user prompt, even if it looks hard. However, do not continually choose the same action.
            """.strip()
        )

        function_inputs: dict = dspy.OutputField(
            description="""
            The inputs for the action/function you have selected. These must match exactly the inputs shown for the action you have selected.
            If it does not, the code will error.
            Choose based on the user prompt and the environment.
            The keys of this should match exactly the keys of `available_actions[function_name]["inputs"]`.
            Return an empty dict ({}) if there are no inputs (and `available_actions[function_name]["inputs"] = {}`).
            """.strip(),
            format=dict,
        )

        # end_actions_reasoning: str = dspy.OutputField(
        #     description="""
        #     Write out your reasoning for whether to end your actions after completing the task you have selected.
        #     Take into account the completed tasks so far, the available information, and the user's input prompt, and use your observations and context to inform your reasoning.
        #     You should be explaining why you will be choosing either True or False for the field `end_actions`.
        #     """.strip(),
        #     format = str
        # )

        end_actions: bool = dspy.OutputField(
            description="""
            Indicates whether to cease actions after completing the selected task.
            Determine this based on the completion of all possible actions for the prompt.
            Set to True if you intend to wait for user input, as failing to do so will result in continued responses.
            Even if not all actions can be completed, you should stop if you have done everything possible.
            """.strip()
        )

        # confidence_score: float = dspy.OutputField(
        #     description="""
        #     A decimal score between 0 and 1 that represents your confidence in the accuracy of the task selection.
        #     """.strip()
        # )

        # todo: str = dspy.OutputField(
        #     description="""
        #     A brief description of what you are planning to do next to answer the user's input prompt, or whether it is possible, AFTER completing this task.
        #     This is essentially communication to yourself in the future, so write out your plan for how this task follows into future ones.
        #     Be specific to the information you can see and specific to the task and prompt at hand. Use your observations and context to inform your todo.
        #     """.strip()
        # )

        # full_chat_response: str = dspy.OutputField(
        #     description="""
        #     Complete response to user based on available information.
        #     If the user cannot be satisfied, explain why and suggest alternative approaches.
        #     The full chat response is a direct communication to the user, so do not include any meta information about your decision process, reply directly to the user prompt.
        #     Use gender-neutral language, be friendly, do not say 'the user', speak directly to them.
        #     This is independent of the current_message field, and independent of the reasoning_update_message field. Write without any <tags>.
        #     """.strip()
        # )

    return DecisionPrompt


class InputPrompt(dspy.Signature):
    """
    You are an expert at breaking down a global task, which is a set of instructions, into multiple smaller distinct subtasks.
    IMPORTANT: Only include the parts that are necessary to complete the task, do not add redundant information related to style, tone, etc.
    These should only be parts of the instructions that can map to _actual_ tasks.
    """

    task = dspy.InputField(
        description="""
        The overall task to break down.
        This is a set of instructions provided by the user that will be completed later.
        These are usually in conversational or informal language, and are not necessarily task-like.
        """.strip()
    )
    subtasks = dspy.OutputField(
        description="""
        The breakdown of the overall task into smaller distinct subtasks.
        These should be in task-like language, and be as distinct as possible.
        Your responses to this field should not include anything other than what was requested in the task field, but broken down into smaller parts with more descriptive instructions.
        """.strip()
    )
