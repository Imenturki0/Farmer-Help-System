
from pydantic import BaseModel

class Question(BaseModel):
    text: str
    mode: str = "general"
    language: str = "en"
    lat: float
    lon: float

