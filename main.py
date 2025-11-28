import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")


# Load the saved encoder and model
project_path = os.path.dirname(os.path.abspath(__file__))

encoder_path = os.path.join(project_path, "model", "encoder.pkl")
encoder = load_model(encoder_path)

model_path = os.path.join(project_path, "model", "model.pkl")
model = load_model(model_path)


# Create a RESTful API using FastAPI
app = FastAPI()


# Create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    """Welcome message endpoint."""
    return {"message": "Welcome to the income prediction API."}


# Create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    """
    Take input data, preprocess it with the trained encoder,
    run model inference, and return the predicted income label.
    """
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process input using the existing encoder
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
    )

    # Run model inference
    _inference = inference(model, data_processed)

    # Map numeric predictions to human-readable labels
    return {"result": apply_label(_inference)}
