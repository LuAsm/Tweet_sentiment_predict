import urllib.request

import cloudpickle
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(url: str) -> Pipeline:
    #file = urllib.request.urlopen(url)
    model = cloudpickle.load(open("./tweet_model.pkl", "rb"))

    #if not model_loaded_correctly(model):
    #  raise Exception("Missmatch in predictions detected!")

    return model


def model_loaded_correctly(model: Pipeline) -> bool:
    sample_input = {
        "Tweet" : "RT @ChristopherJM: Zelensky also confirms CIA Director Bill Burns' Kyiv visit Tuesday, during Russia's missile attack. “Yesterday Burns sat…",
        
        
    }
    output: str = model.predict(pd.DataFrame([sample_input]))[0]
    return output == "Negative"