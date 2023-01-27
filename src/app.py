from typing import Dict, List
from src.db.connection import mydb
import numpy as np
import pandas as pd
from fastapi import FastAPI
from numpy.typing import NDArray

from src.models import tweet
from src.utils.model_utils import load_model

#Loading model
tweet_model= load_model("https://www.dropbox.com/s/b7dztoz3iry3suw/tweet_model.pkl?dl=0")

class DbConnection:
    pass


def write_to_db(log: str) -> bool:
    db = DbConnection()
    try:
        db.write(log)
        print("DB write ok")
    except:
        print("DB write failed")
        return False
    return True


app = FastAPI()



@app.get("/health")
def health() -> Dict[str, str]:
    return {"health": "alive"}



@app.post("/predict")
def predict_tweet(inputs: List[tweet.Input]) -> Dict[str, List[Dict[str, float]]]:


    parsed_input = pd.DataFrame([i.dict() for i in inputs])
    outputs: NDArray[np.float32] = tweet_model.predict(parsed_input["Tweet"].values)
    logs = {"path": "/predict", "input": input, "preds": outputs}
    print (logs)
    write_to_db(logs)

    return {"outputs": [{"Sentiment": i} for i in outputs.tolist()]}
