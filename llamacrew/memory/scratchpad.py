"""Shared scratchpad memory for agents."""

from typing import Any, Dict, Optional, Protocol


class MemoryBackend(Protocol):
    """Protocol for memory backends."""

    def set(self, key: str, value: Any) -> None:
        """Store a value."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a key."""
        ...

    def get_all(self) -> Dict[str, Any]:
        """Get all key-value pairs."""
        ...

    def keys(self) -> list:
        """Get all keys."""
        ...

    def clear(self) -> None:
        """Clear all data."""
        ...


class Scratchpad:
    """
    Shared memory scratchpad for agents to read/write data.

    This provides a simple key-value store that all agents in a crew can access.
    Can use either in-memory storage or a persistent backend.
    """

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        """
        Initialize scratchpad.

        Args:
            backend: Optional memory backend (e.g., LlamaStackMemoryBackend).
                     If None, uses in-memory storage.
        """
        self._backend = backend
        self._data: Dict[str, Any] = {}  # Used if no backend provided

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the scratchpad.

        Args:
            key: Key to store under
            value: Value to store
        """
        if self._backend:
            self._backend.set(key, value)
        else:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the scratchpad.

        Args:
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value for the key, or default if not found
        """
        if self._backend:
            return self._backend.get(key, default)
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        """
        Delete a key from the scratchpad.

        Args:
            key: Key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        if self._backend:
            return self._backend.delete(key)

        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all data from the scratchpad."""
        if self._backend:
            self._backend.clear()
        else:
            self._data.clear()

    def get_all(self) -> Dict[str, Any]:
        """
        Get all data from the scratchpad.

        Returns:
            Copy of all data
        """
        if self._backend:
            return self._backend.get_all()
        return self._data.copy()

    def keys(self) -> list:
        """Get all keys in the scratchpad."""
        if self._backend:
            return self._backend.keys()
        return list(self._data.keys())

    def has_key(self, key: str) -> bool:
        """Check if a key exists in the scratchpad."""
        if self._backend:
            value = self._backend.get(key)
            return value is not None
        return key in self._data

    def to_dict(self) -> Dict[str, Any]:
        """Export scratchpad to dictionary."""
        return self._data.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scratchpad":
        """
        Create scratchpad from dictionary.

        Args:
            data: Dictionary to load

        Returns:
            Scratchpad instance
        """
        scratchpad = cls()
        scratchpad._data = data.copy()
        return scratchpad

    def __repr__(self) -> str:
        """String representation."""
        return f"Scratchpad(keys={list(self._data.keys())})"
