import cProfile
import pstats
import atexit
from functools import wraps


def persistent_profile(output_file="profile.prof", sort_by="cumulative", lines=5):
    """
    Decorator that accumulates profiling data across multiple calls
    and writes combined results to a .prof file at program using the at exit handler.
    """
    profiler = cProfile.Profile()
    printed = False

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            return result

        def dump_profile():
            nonlocal printed
            if printed:
                return
            printed = True
            print(f"\n[Saving function call profiling data to {output_file}]")
            profiler.dump_stats(output_file)
            ps = pstats.Stats(profiler).sort_stats(sort_by)
            ps.print_stats(lines)

        atexit.register(dump_stats)
        return wrapper
    return decorator

