from datetime import datetime
from functools import wraps

def print_timestamp(message=""):
    """
    Print a timestamp with an optional message.
    """
    print(f"{message} Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def timestamp_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed {func.__name__}")
        return result
    return wrapper
