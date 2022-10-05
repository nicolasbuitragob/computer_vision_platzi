from fastapi import FastAPI
from .prediction import CountModel
from .models import PredictionRequest

app = FastAPI(docs_url='/docs')


@app.post('/car-motorbike-counter')
def parse(request: PredictionRequest):

    #extract encoded string
    videoBase64 = request.video

    #init class
    sc = CountModel()
    
    #predict video using string
    response_64 = sc.predict(videoBase64)
    
    #returns video base64
    return response_64