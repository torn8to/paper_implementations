import cProfile
import pstats
import atexit
from functools import wraps


def persistent_profile(starts_after: int = 400, output_file: str = "profile.prof", sort_by: str = "time", lines: int = 5):
    """
    Decorator that accumulates profiling data across multiple calls
    and writes combined results to a .prof file at program using the at exit handler.
    """
    call_counter: dict = {"counter": 0}
    call_threshold: int = starts_after
    profiler: cProfile.Profile = cProfile.Profile()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal call_counter
            nonlocal call_threshold

            call_counter["counter"] = call_counter["counter"] + 1
            if call_counter["counter"] > call_threshold:
                profiler.enable()
            result = func(*args, **kwargs)
            if call_counter["counter"] > call_threshold:
                profiler.disable()
            return result

        def dump_profile():
            print(f"\n[Saving function call profiling data to {output_file}]")
            profiler.dump_stats(output_file)
            ps = pstats.Stats(profiler).sort_stats(sort_by)
            ps.print_stats(lines)
            return

        # on program end save this shit
        atexit.register(dump_profile)
        return wrapper

    return decorator
