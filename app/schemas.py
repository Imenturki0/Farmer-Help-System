
from pydantic import BaseModel
from typing import Optional

class Question(BaseModel):
    text: str
    session_id: str = "default"
    mode: str = "general"
    language: str = "en"
    lat: Optional[float] = None
    lon: Optional[float] = None

