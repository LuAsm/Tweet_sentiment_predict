from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from numpy.typing import NDArray

from src.models import tweet
from src.utils.model_utils import load_model

#Loading model
tweet_model= load_model("https://www.dropbox.com/s/b7dztoz3iry3suw/tweet_model.pkl?dl=0")

app = FastAPI()



@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "alive"}


@app.post("/predict_tweet_Sentiment")
def predict_tweet(inputs: List[tweet.Input]) -> Dict[str, List[Dict[str, float]]]:
    parsed_input = pd.DataFrame([i.dict() for i in inputs])
    outputs: NDArray[np.float32] = tweet_model.predict(parsed_input["Tweet"].values)

    return {"outputs": [{"Sentiment": i} for i in outputs.tolist()]}
