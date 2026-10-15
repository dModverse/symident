"""Every test runs with the DEBUG switch: an exception that a route would record in
info["errors"] fails the test instead."""

import os

os.environ.setdefault("SYMIDENT_DEBUG", "1")
