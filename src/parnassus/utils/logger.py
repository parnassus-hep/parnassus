from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from io import TextIOWrapper

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

is_terminal = Console().is_terminal
type ProgressFactory = Callable[[], Progress]

ProgressBar: ProgressFactory = partial(
    Progress,
    TextColumn("[task.description]{task.description}"),
    BarColumn(),
    TextColumn("{task.completed}/{task.total}"),
    TextColumn("•"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("•"),
    TimeRemainingColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    refresh_per_second=1 if is_terminal else 0.05,
    speed_estimate_period=30 if is_terminal else 120,
    console=Console(width=100, force_terminal=True),
)


def update_task(progress_bar: Progress, task: TaskID) -> Callable[[], None]:
    """Create a function to update a progress bar task.

    Parameters
    ----------
    progress_bar : Progress
        Progress bar instance.
    task : TaskID
        Task ID to update.

    Returns
    -------
    Callable[[], None]
        Function that updates the progress bar task when called.
    """

    def update():
        progress_bar.update(task, advance=1)

    return update


def setup_logger(level: str = "INFO"):
    """Setup rich logger for pretty console output.

    Parameters
    ----------
    level : str
        Logging level (e.g., "INFO", "DEBUG").

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    FORMAT = "%(message)s"  # noqa: N806
    console = None if is_terminal else Console(width=120)
    handler = RichHandler(
        show_time=False, show_path=False, markup=True, rich_tracebacks=True, console=console
    )
    logging.basicConfig(level=level, format=FORMAT, datefmt="[%X]", handlers=[handler])
    return logging


@contextmanager
def stdout_redirected(to: str = os.devnull):
    """Redirect stdout to suppress C++ library output (e.g., fastjet).

    Parameters
    ----------
    to : str
        File path to redirect stdout to. Defaults to os.devnull.

    Examples
    --------
    >>> with stdout_redirected(to=filename):
    ...     print("from Python")
    ...     os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to: TextIOWrapper):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout
