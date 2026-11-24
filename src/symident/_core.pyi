# The pybind11 extension built from cpp/; the functions are untyped.
from typing import Any

def __getattr__(name: str) -> Any: ...
