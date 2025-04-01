from pydantic import BaseModel

class CreateAccountDto(BaseModel):
    firstname: str
    username: str
    password: str