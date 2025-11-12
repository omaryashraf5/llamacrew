"""Llama Stack native backend for memory using API endpoints."""

import json
from typing import Any, Dict, List, Optional

from llama_stack_client import LlamaStackClient


class LlamaStackMemoryBackend:
    """
    Memory backend using Llama Stack native endpoints.

    Uses /v1/conversations for key-value storage and conversational memory.
    This provides a simpler, integrated approach without external dependencies.
    """

    def __init__(
        self,
        client: LlamaStackClient,
        conversation_id: Optional[str] = None,
        crew_id: Optional[str] = None,
    ):
        """
        Initialize Llama Stack backend.

        Args:
            client: Llama Stack client instance
            conversation_id: Optional conversation ID to use for memory
            crew_id: Optional crew ID for metadata
        """
        self.client = client
        self.conversation_id = conversation_id
        self.crew_id = crew_id
        self._cache: Dict[str, Any] = {}  # Local cache for performance

        # Create conversation if not provided
        if not self.conversation_id:
            self.conversation_id = self._create_crew_conversation()

    def _create_crew_conversation(self) -> str:
        """
        Create a crew-level conversation for shared memory.

        Returns:
            Conversation ID
        """
        metadata = {"type": "crew_memory"}
        if self.crew_id:
            metadata["crew_id"] = self.crew_id

        try:
            # Create conversation using the conversations API
            response = self.client.post("/v1/conversations", json={"metadata": metadata})
            return response.json()["id"]
        except Exception:
            # Fallback: use a generated ID
            import uuid

            conv_id = f"crew_memory_{uuid.uuid4().hex[:8]}"
            self._cache["_conversation_id"] = conv_id
            return conv_id

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in memory.

        Args:
            key: Storage key
            value: Value to store (will be JSON serialized)
        """
        # Update local cache
        self._cache[key] = value

        # Store in conversation as a special message
        try:
            serialized = json.dumps(value)
            self.client.post(
                f"/v1/conversations/{self.conversation_id}/messages",
                json={"role": "system", "content": f"MEMORY_SET:{key}={serialized}"},
            )
        except Exception:
            # If API call fails, keep in local cache
            pass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from memory.

        Args:
            key: Storage key
            default: Default value if key not found

        Returns:
            Stored value or default
        """
        # Check local cache first
        if key in self._cache:
            return self._cache[key]

        # Try to load from conversation history
        try:
            response = self.client.get(f"/v1/conversations/{self.conversation_id}")
            messages = response.json().get("messages", [])

            # Search for most recent MEMORY_SET for this key
            for msg in reversed(messages):
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if content.startswith(f"MEMORY_SET:{key}="):
                        value_str = content[len(f"MEMORY_SET:{key}=") :]
                        value = json.loads(value_str)
                        self._cache[key] = value
                        return value
        except Exception:
            pass

        return default

    def delete(self, key: str) -> bool:
        """
        Delete a key from memory.

        Args:
            key: Storage key

        Returns:
            True if key existed and was deleted
        """
        existed = key in self._cache
        if existed:
            del self._cache[key]

        # Mark as deleted in conversation
        try:
            self.client.post(
                f"/v1/conversations/{self.conversation_id}/messages",
                json={"role": "system", "content": f"MEMORY_DELETE:{key}"},
            )
        except Exception:
            pass

        return existed

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        if key in self._cache:
            return True

        # Check conversation history
        value = self.get(key)
        return value is not None

    def keys(self, pattern: str = "*") -> List[str]:
        """
        Get all keys matching pattern.

        Args:
            pattern: Key pattern (supports * wildcard)

        Returns:
            List of matching keys
        """
        # For now, return cached keys
        # Could be enhanced to parse conversation history
        all_keys = list(self._cache.keys())

        if pattern == "*":
            return all_keys

        # Simple pattern matching
        import fnmatch

        return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]

    def get_all(self) -> Dict[str, Any]:
        """
        Get all key-value pairs.

        Returns:
            Dictionary of all data
        """
        return self._cache.copy()

    def clear(self, pattern: str = "*") -> int:
        """
        Clear keys matching pattern.

        Args:
            pattern: Key pattern to clear

        Returns:
            Number of keys deleted
        """
        keys_to_delete = self.keys(pattern)
        count = 0

        for key in keys_to_delete:
            if self.delete(key):
                count += 1

        return count

    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: Amount to increment by

        Returns:
            New counter value
        """
        current = self.get(key, 0)
        new_value = current + amount
        self.set(key, new_value)
        return new_value

    def append_to_list(self, key: str, value: Any) -> None:
        """
        Append value to a list.

        Args:
            key: List key
            value: Value to append
        """
        current_list = self.get(key, [])
        if not isinstance(current_list, list):
            current_list = []
        current_list.append(value)
        self.set(key, current_list)

    def get_list(self, key: str) -> List[Any]:
        """
        Get entire list.

        Args:
            key: List key

        Returns:
            List of values
        """
        value = self.get(key, [])
        return value if isinstance(value, list) else []

    def close(self) -> None:
        """Close backend (no-op for Llama Stack)."""
        pass

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()
        return False


class VectorStoreBackend:
    """
    Backend for vector-based semantic memory using /v1/vector_stores.

    Provides semantic search and RAG capabilities for agents.
    """

    def __init__(
        self,
        client: LlamaStackClient,
        vector_store_id: Optional[str] = None,
        embedding_model: str = "sentence-transformers",
        embedding_dimension: int = 384,
    ):
        """
        Initialize vector store backend.

        Args:
            client: Llama Stack client instance
            vector_store_id: Optional vector store ID
            embedding_model: Model to use for embeddings
            embedding_dimension: Dimension of embeddings
        """
        self.client = client
        self.vector_store_id = vector_store_id
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        # Create vector store if not provided
        if not self.vector_store_id:
            self.vector_store_id = self._create_vector_store()

    def _create_vector_store(self) -> str:
        """
        Create a new vector store.

        Returns:
            Vector store ID
        """
        try:
            response = self.client.post(
                "/v1/vector_stores",
                json={
                    "embedding_model": self.embedding_model,
                    "embedding_dimension": self.embedding_dimension,
                },
            )
            return response.json()["id"]
        except Exception:
            # Fallback: use a generated ID
            import uuid

            return f"vector_store_{uuid.uuid4().hex[:8]}"

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add text to vector store.

        Args:
            text: Text to add
            metadata: Optional metadata

        Returns:
            Document ID
        """
        try:
            response = self.client.post(
                f"/v1/vector_stores/{self.vector_store_id}/documents",
                json={"content": text, "metadata": metadata or {}},
            )
            return response.json()["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to add text to vector store: {e}") from e

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search vector store for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        try:
            response = self.client.post(
                f"/v1/vector_stores/{self.vector_store_id}/search",
                json={"query": query, "top_k": top_k},
            )
            return response.json().get("results", [])
        except Exception as e:
            raise RuntimeError(f"Failed to search vector store: {e}") from e

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from vector store.

        Args:
            document_id: Document ID

        Returns:
            True if successful
        """
        try:
            self.client.delete(f"/v1/vector_stores/{self.vector_store_id}/documents/{document_id}")
            return True
        except Exception:
            return False

    def clear(self) -> bool:
        """
        Clear all documents from vector store.

        Returns:
            True if successful
        """
        try:
            self.client.delete(f"/v1/vector_stores/{self.vector_store_id}")
            # Recreate empty store
            self.vector_store_id = self._create_vector_store()
            return True
        except Exception:
            return False


class FileStorageBackend:
    """
    Backend for file storage using /v1/files.

    Handles document uploads, downloads, and management.
    """

    def __init__(self, client: LlamaStackClient):
        """
        Initialize file storage backend.

        Args:
            client: Llama Stack client instance
        """
        self.client = client

    def upload_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Upload a file.

        Args:
            file_path: Path to file to upload
            metadata: Optional metadata

        Returns:
            File ID
        """
        try:
            with open(file_path, "rb") as f:
                files = {"file": f}
                data = {"metadata": json.dumps(metadata or {})}

                response = self.client.post("/v1/files", files=files, data=data)
                return response.json()["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to upload file: {e}") from e

    def upload_content(
        self, content: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Upload file content directly.

        Args:
            content: File content as bytes
            filename: Name for the file
            metadata: Optional metadata

        Returns:
            File ID
        """
        try:
            files = {"file": (filename, content)}
            data = {"metadata": json.dumps(metadata or {})}

            response = self.client.post("/v1/files", files=files, data=data)
            return response.json()["id"]
        except Exception as e:
            raise RuntimeError(f"Failed to upload content: {e}") from e

    def download_file(self, file_id: str, destination: str) -> None:
        """
        Download a file.

        Args:
            file_id: File ID
            destination: Path to save file
        """
        try:
            response = self.client.get(f"/v1/files/{file_id}/content")
            with open(destination, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise RuntimeError(f"Failed to download file: {e}") from e

    def get_file_content(self, file_id: str) -> bytes:
        """
        Get file content as bytes.

        Args:
            file_id: File ID

        Returns:
            File content
        """
        try:
            response = self.client.get(f"/v1/files/{file_id}/content")
            return response.content
        except Exception as e:
            raise RuntimeError(f"Failed to get file content: {e}") from e

    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get file metadata.

        Args:
            file_id: File ID

        Returns:
            File metadata
        """
        try:
            response = self.client.get(f"/v1/files/{file_id}")
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to get file metadata: {e}") from e

    def list_files(self) -> List[Dict[str, Any]]:
        """
        List all files.

        Returns:
            List of file metadata
        """
        try:
            response = self.client.get("/v1/files")
            return response.json().get("files", [])
        except Exception as e:
            raise RuntimeError(f"Failed to list files: {e}") from e

    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file.

        Args:
            file_id: File ID

        Returns:
            True if successful
        """
        try:
            self.client.delete(f"/v1/files/{file_id}")
            return True
        except Exception:
            return False
