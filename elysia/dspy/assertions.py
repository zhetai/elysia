import inspect

import dspy
from dspy import Module, Signature


class AssertionError(Exception):
    def __init__(
        self, feedback: str, prediction: dspy.Prediction, target_module: Module
    ):
        self.feedback = feedback
        self.prediction = prediction
        self.target_module = target_module

    def __str__(self):
        return self.feedback


def Assert(
    condition: bool, feedback: str, prediction: dspy.Prediction, target_module: Module
):
    if not condition:
        raise AssertionError(feedback, prediction, target_module)


class AssertedModule(Module):
    def __init__(self, module: Module, max_tries: int = 3):
        super().__init__()
        self.module = module
        self.max_tries = max_tries
        self.tries = 0
        self.original_module = None

    def modify_module(self, module: Module, prediction: dspy.Prediction):
        self.original_module = module.deepcopy()
        signature: Signature = module.predict.signature
        signature = signature.append(
            "feedback",
            dspy.InputField(
                desc="Feedback from the previous error",
                type_=str,
            ),
        )
        for field in prediction.toDict():
            signature = signature.append(
                f"previous_{field}",
                dspy.InputField(
                    desc=(
                        f"The previous input to {field} which caused the error. "
                        "It is not necessarily this field which caused the error, "
                        "if there are other previous fields that were previous inputs."
                    ),
                    type_=type(prediction[field]),
                ),
            )

        module.predict.signature = signature
        return module

    def reset_module(self, module: Module):
        if self.original_module:
            module = self.original_module.deepcopy()

    def __call__(self, *args, **kwargs):
        e_copy = None
        while self.tries < self.max_tries:
            try:
                return self.module(*args, **kwargs)
            except AssertionError as e:
                if self.tries == self.max_tries:
                    raise e

                self.modify_module(
                    e.target_module,
                    e.prediction,
                )
                kwargs["feedback"] = e.feedback
                for field in e.prediction.toDict():
                    kwargs[f"previous_{field}"] = e.prediction[field]
                self.tries += 1
                e_copy = e
            finally:
                if e_copy:
                    self.reset_module(e_copy.target_module)
                self.tries = 0


if __name__ == "__main__":
    lm = dspy.LM("gpt-4o-mini")
    dspy.configure(lm=lm)
    # mod1 = dspy.ChainOfThought("x -> y")
    # pred = mod1(x="what is the capital of spain?")
    # print(pred)

    class TestModule(Module):
        def __init__(self):
            super().__init__()
            self.prompt = dspy.ChainOfThought("x -> y")

        def __call__(self, x: str, **kwargs):
            out = self.prompt(x=x, **kwargs)
            Assert(
                len(out.y) > 25,
                "Output is too short. Provide a more detailed answer.",
                out,
                self.prompt,
            )
            return out.y

    # mod1 = TestModule()
    # pred = mod1(x="what is the capital of spain?")
    # print(pred)

    mod2 = AssertedModule(TestModule())
    pred = mod2(x="what is the capital of uk?")
    print(pred)
