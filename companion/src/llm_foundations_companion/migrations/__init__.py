"""SQLite migration inventory for the local companion."""

from .v001 import CORE_SCHEMA_VERSION, CORE_STATEMENTS

__all__ = ["CORE_SCHEMA_VERSION", "CORE_STATEMENTS"]
