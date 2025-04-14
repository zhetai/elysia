import inspect
import os
import traceback

import psutil
from rich import print
from rich.panel import Panel


def create_error_log(error_message, fname="error_log.txt"):
    if os.path.exists(fname):
        with open(fname, "a") as f:
            f.write("ERROR: " + error_message + "\n" + traceback.format_exc())
    else:
        with open(fname, "w") as f:
            f.write("ERROR: " + error_message + "\n" + traceback.format_exc())


def backend_print(*args, **kwargs):
    for arg in args:
        print(f"[bold]BACKEND:[/bold]  {arg}", **kwargs)


def backend_print_warning(*args, **kwargs):
    for arg in args:
        print(f"[bold orange]BACKEND WARNING:[/bold orange]  {arg}", **kwargs)


def backend_print_panel(text, **kwargs):
    print(
        Panel.fit(
            text,
            title=kwargs.get("title", ""),
            padding=(1, 1),
            border_style=kwargs.get("border_style", "yellow"),
        )
    )


def backend_print_error(error_message, **kwargs):
    # try:
    #     line_no = inspect.currentframe().f_lineno
    # except:
    line_no = "unknown"
    print(
        f"\n\n[bold red]BACKEND ERROR:[/bold red] [red]Line {line_no}[/red] {error_message}\n\n",
        **kwargs,
    )
    print(f"TRACEBACK: {traceback.format_exc()}")
    create_error_log(error_message)


def print_memory_usage(name=""):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    if name != "":
        name = f" At [italic]{name}[/italic]:"
    backend_print(
        f"[bold green]Memory usage:[/bold green]{name} {memory_info.rss / 1024 / 1024:.2f} MB"
    )
