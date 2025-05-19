import threading
import time
from typing import Callable, Tuple

from src.utils.constants import (
    MAX_INFERENCE_RETRIES,
    SEND_REQUEST_TIMEOUT,
    TIME_BETWEEN_RATE_LIMIT_RETRIES,
)
from src.utils.custom_exceptions import (
    FunctionCallTimeoutError,
    OpenAIAPICallError,
)

def attempt_function_call(
    func: Callable, max_attempts: int = MAX_INFERENCE_RETRIES, *args, **kwargs
) -> Tuple[object, int, float]:
    """
    Attempt to call a function and return its result.

    Parameters
    ----------
    func: function
        The function to call.
    max_attempts: int
        The maximum number of attempts to call the function.
    *args: list
        The positional arguments to pass to the function.
    **kwargs: dict
        The keyword arguments to pass to the function.

    Raises
    ------
    RateLimitError:
        If the OpenAI API rate limit is exceeded after the maximum number of attempts.
    Exception:
        If the function call fails.

    Returns
    -------
    Tuple[object, int, float]:
        The result of the function call, the number of attempts, and the time (s) it
        took to execute the function including all attempts.
    """
    num_attempts = 0

    time_start = time.time()

    for _ in range(max_attempts):
        num_attempts += 1
        print(f"Attempt {num_attempts} of {max_attempts}...")

        try:
            result = call_function_with_timeout(
                func, timeout=SEND_REQUEST_TIMEOUT, *args, **kwargs
            )
            break

        except Exception as e:
            if num_attempts == max_attempts:
                raise e
            elif (
                isinstance(e, OpenAIAPICallError)
                and e.openai_error == "RateLimitError"
            ):
                # If the rate limit is exceeded, wait for a specified time before retrying
                print(
                    f"API RATE LIMIT EXCEEDED!! Retrying in {TIME_BETWEEN_RATE_LIMIT_RETRIES} seconds..."
                )
                time.sleep(TIME_BETWEEN_RATE_LIMIT_RETRIES)


    return result, num_attempts, time.time() - time_start


def call_function_with_timeout(func: Callable, timeout: int, *args, **kwargs) -> object:
    """
    Call a function with a timeout.

    Parameters
    ----------
    func: function
        The function to call.
    timeout: int
        The timeout in seconds.
    *args: list
        The positional arguments to pass to the function.
    **kwargs: dict
        The keyword arguments to pass to the function.

    Raises
    ------
    FunctionCallTimeoutError:
        If the function call times out.
    Exception:
        If the function call fails.

    Returns
    -------
    object:
        The result of the function call.
    """
    result = [None]
    exception = [None]

    def wrapper():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise FunctionCallTimeoutError()

    if exception[0]:
        raise exception[0]

    return result[0]
