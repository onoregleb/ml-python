from pydantic import BaseModel
from typing import Union, List, Dict, Any

class CreatePredictionDto(BaseModel):
    model_id: str
    input_data: Union[Dict[str, float], List[Dict[str, float]]]