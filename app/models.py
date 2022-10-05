from pydantic import BaseModel

class PredictionRequest(BaseModel):
    video:str
    

class PredictionResponse(BaseModel):
    video:str