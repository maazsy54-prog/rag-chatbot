"""Exceptions raised by the ingestion pipeline.

The API layer maps each one to an HTTP status code, so business logic never
needs to know about HTTP.
"""


class IngestionError(Exception):
    """Base class for every ingestion failure."""


class UnsupportedFileTypeError(IngestionError):
    """The file's real type is not supported (HTTP 415)."""


class EmptyDocumentError(IngestionError):
    """The file has no extractable text, e.g. it is empty or scanned (HTTP 400)."""


class CorruptDocumentError(IngestionError):
    """The file is damaged, encrypted or otherwise unreadable (HTTP 400)."""


class DocumentTooLargeError(IngestionError):
    """The file or document exceeds the configured size limits (HTTP 413)."""


class DuplicateDocumentError(IngestionError):
    """The same file was already uploaded by this user (HTTP 409).

    Attributes:
        document_id: Id of the document that already holds this content.
    """

    def __init__(self, document_id: int) -> None:
        super().__init__(f"This document was already uploaded (id {document_id}).")
        self.document_id = document_id


class EmbeddingServiceError(IngestionError):
    """The embedding API failed, timed out or returned unusable data (HTTP 503)."""


class DatabaseUnavailableError(IngestionError):
    """The database cannot be reached or timed out (HTTP 503)."""
