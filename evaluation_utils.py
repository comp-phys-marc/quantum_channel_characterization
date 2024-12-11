import time
from functools import wraps, partial


def profile(func):
    """
    Provides automatic time profiling.

    :param func: A method that we intend to profile.
    :return: The wrapped method with profiling added.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        pfunc = partial(func, *args, **kwargs)
        func_name = func.__name__
        start_time = time.time()
        result = pfunc()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func_name}: {elapsed_time * 1000} ms")
        return result
    return wrapper


if __name__ == "__main__":
    @profile
    def wait():
        time.sleep(5)

    wait()
