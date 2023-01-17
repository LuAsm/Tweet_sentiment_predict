from pydantic import BaseModel


class TweetInput(BaseModel):
    Tweet : str
    polarity: float
    
    