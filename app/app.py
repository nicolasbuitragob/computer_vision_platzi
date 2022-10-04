from urllib import request
from fastapi import FastAPI,Request
from .prediction import CountModel
from .models import PredictionRequest

app = FastAPI(docs_url='/docs')


@app.post('/counter')
def parse(request:PredictionRequest):
    
    videoBase64 = request
    print(request)
    sc = CountModel()
    
    response_64 = sc.predict(videoBase64)
    output=response_64
    return 'ok'