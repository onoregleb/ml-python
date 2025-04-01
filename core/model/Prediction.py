from typing import Optional, Dict, Any
from datetime import datetime


class Prediction:
    def __init__(self, id: str, model_id: str, user_id: str, input_data: Dict[str, Any],
                 output_data: Optional[Dict[str, Any]] = None, status: str = "pending"):
        self.id = id
        self.model_id = model_id
        self.user_id = user_id
        self.input_data = input_data
        self.output_data = output_data
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def get_id(self):
        return self.id

    def get_model_id(self):
        return self.model_id

    def get_user_id(self):
        return self.user_id

    def get_input_data(self):
        return self.input_data

    def get_output_data(self):
        return self.output_data

    def get_status(self):
        return self.status

    def set_output_data(self, output_data: Dict[str, Any]):
        self.output_data = output_data
        self.updated_at = datetime.now()
        self.status = "completed"

    def set_failed(self):
        self.status = "failed"
        self.updated_at = datetime.now()