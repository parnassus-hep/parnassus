"""Execution utilities for pipeline processing with configurable parallelization."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from parnassus.configs.scheme import GenEvent
from parnassus.utils.logger import ProgressBar


def _worker_wrapper[T](worker_fn: Callable[[list[Any], Any], T], args: tuple[list[Any], Any]) -> T:
    """Module-level wrapper to unpack arguments for pool.imap.

    This must be at module level (not nested) to be picklable by multiprocessing.

    Parameters
    ----------
    worker_fn : Callable[[list[Any], Any], T]
        The worker function to call.
    args : tuple[list[Any], Any]
        Tuple of (batch_input, config) to unpack.

    Returns
    -------
    T
        Result from worker function.
    """
    batch_input, cfg = args
    return worker_fn(batch_input, cfg)


def process_batches[T](
    events: Sequence[GenEvent],
    config: Any,  # Should have batch_size and num_processes attributes
    worker_fn: Callable[[list[Any], Any], T],
    extract_fn: Callable[[Sequence[GenEvent], range, Any], list[Any]],
    description: str,
) -> list[T]:
    """Process events in batches with configurable parallelization.

    This function provides a unified interface for batch processing that automatically
    switches between synchronous (single-process) and parallel (multi-process) execution
    based on the configuration. When num_processes=1, it avoids the overhead of
    multiprocessing and pickling by running workers synchronously.

    Parameters
    ----------
    events : Sequence[GenEvent]
        Events to process.
    config : Any
        Configuration object with batch_size and num_processes attributes.
    worker_fn : Callable[[list[Any], Any], T]
        Worker function that processes a batch of extracted data.
        Signature: worker_fn(batch_data: list[Any], config: Any) -> T
    extract_fn : Callable[[Sequence[GenEvent], range, Any], list[Any]]
        Function to extract lightweight data from events for a batch.
        Signature: extract_fn(events: Sequence[GenEvent], batch_indices: range,
        config: Any) -> list[Any]
    description : str
        Description for progress bar.

    Returns
    -------
    list[T]
        List of results from worker function, one per batch.

    Raises
    ------
    RuntimeError
        Re-raises any exception from worker function with batch context added.

    Examples
    --------
    >>> def extract_data(events, batch_indices, config):
    ...     return [{"pt": events[i].particles.pt} for i in batch_indices]
    >>> def worker(batch_data, config):
    ...     return [sum(d["pt"]) for d in batch_data]
    >>> results = process_batches(events, config, worker, extract_data, "Processing")
    """
    n_events = len(events)
    batch_size = config.batch_size
    num_processes = config.num_processes

    # Calculate number of batches
    n_batches = n_events // batch_size
    n_batches += 1 if n_events % batch_size != 0 else 0

    # Extract batched data using the provided extract function
    batched_data: list[tuple[list[Any], Any]] = []
    for i in range(n_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, n_events)
        batch_indices = range(batch_start, batch_end)
        extracted = extract_fn(events, batch_indices, config)
        batched_data.append((extracted, config))

    results: list[T] = []

    if num_processes <= 1:
        # Synchronous execution - avoid multiprocessing overhead
        with ProgressBar() as progress:
            task = progress.add_task(f"[green]{description}", total=n_events)
            for i, (batch_input, cfg) in enumerate(batched_data):
                try:
                    result = worker_fn(batch_input, cfg)
                    results.append(result)
                    progress.update(task, advance=len(batch_input))
                except Exception as e:
                    batch_start = i * batch_size
                    batch_end = min((i + 1) * batch_size, n_events)
                    raise RuntimeError(
                        f"Error processing batch {i} (events {batch_start}-{batch_end}): {e}"
                    ) from e
    else:
        # Parallel execution using multiprocessing
        # Use partial to bind worker_fn to the module-level wrapper
        wrapper = partial(_worker_wrapper, worker_fn)

        with mp.Pool(processes=num_processes) as pool, ProgressBar() as progress:
            task = progress.add_task(f"[green]{description}", total=n_events)
            for i, result in enumerate(pool.imap(wrapper, batched_data)):
                results.append(result)
                batch_start = i * batch_size
                batch_end = min((i + 1) * batch_size, n_events)
                progress.update(task, advance=batch_end - batch_start)

    return results
