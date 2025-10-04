from typing import Optional, Any
from enum import Enum
from pydantic import BaseModel

class ChunkType(str, Enum):
    ALL = "all"
    TECHNICAL = "technical"
    LEGAL = "legal"
    ADDITIONAL = "additional"
    NOTICE = "notice"

class ProcurementDocument(BaseModel):
    """Simple document schema for data retrieval"""
    title: str
    document_type: str
    access_url: Optional[str] = None

class TenderChunks(BaseModel):
    """Response for chunks endpoints with standardized structure."""
    chunks: Any

class TenderChunksResponse(BaseModel):
    """Standardized response model for chunk operations"""
    success: bool
    message: Optional[str] = None
    data: Optional[TenderChunks] = None
