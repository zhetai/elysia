import inspect
import os
import traceback

import psutil
from rich import print
from rich.panel import Panel


# def create_error_log(error_message, fname="error_log.txt"):
#     if os.path.exists(fname):
#         with open(fname, "a") as f:
#             f.write("ERROR: " + error_message + "\n" + traceback.format_exc())
#     else:
#         with open(fname, "w") as f:
#             f.write("ERROR: " + error_message + "\n" + traceback.format_exc())


# def print_panel(text, **kwargs):
#     print(
#         Panel.fit(
#             text,
#             title=kwargs.get("title", ""),
#             padding=(1, 1),
#             border_style=kwargs.get("border_style", "yellow"),
#         )
#     )


def print_memory_usage(name=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if name != "":
        name = f" At [italic]{name}[/italic]:"
    print(
        f"[bold green]Memory usage:[/bold green]{name} {memory_info.rss / 1024 / 1024:.2f} MB"
    )
