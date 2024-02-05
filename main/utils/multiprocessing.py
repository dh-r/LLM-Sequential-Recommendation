from typing import Callable, Any
from multiprocessing.pool import ThreadPool


def execute_function_on_threads(
    num_threads: int,
    function: Callable[..., Any],
    function_args: list,
    results_aggregate_init: Any,
    results_aggregate_function: Callable[..., Any],
) -> Any:
    """Execute a function on multiple threads using a thread pool and aggregate
        the results.

    Args:
        num_threads (int): The number of threads to use for executing the function.
        function (Callable[..., Any]): The function to execute.
        function_args (list): A list of list containing the arguments for the function.
        results_aggregate_init (Any): The initial value for the results aggregate.
        results_aggregate_function (Callable[..., Any]): A function that aggregates
            the results.

    Returns:
        Any: The final results aggregate.
    """
    with ThreadPool(processes=num_threads) as pool:
        results = []
        for i in range(num_threads):
            results.append(
                pool.apply_async(
                    function,
                    function_args[i] if len(function_args) > i else (),
                )
            )

        results_aggregated = results_aggregate_init
        for val in results:
            try:
                results_aggregated = results_aggregate_function(
                    results_aggregated, val.get()
                )
            except Exception as e:
                raise e  # Propagate exceptions to the main process.

        pool.close()
        pool.join()

    return results_aggregated
