from pydantic import BaseModel
from typing import Optional

class CreatePredictionDto(BaseModel):
    model_id: str
    input_data: dict
    user_id: Optional[str] = None