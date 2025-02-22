from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, ValidationError
import xgboost as xgb
import pandas as pd
import joblib
import logging
import os

# Configure logging for tracking events and errors
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Define the input data model for API requests
class InputData(BaseModel):
    latitude: float
    longitude: float
    housing_median_age: float
    population: float
    total_rooms: float
    total_bedrooms: float
    households: float
    median_income: float
    ocean_proximity: str


def load_model():
    """
    This method loads the trained machine learning model from a specified file path. In case of failure, the
    corresponding error is logged.
    """
    try:
        model_path = os.getenv("MODEL_PATH", "Models/CaliforniaBestHousingModel.json")
        model_created = xgb.XGBRegressor()
        model_created.load_model(model_path)
        logging.info("Model loaded successfully.")
        return model_created
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model.")


def load_preprocessor():
    """
    This method is responsible for loading the preprocessor used for the transformation of the features.
    """
    try:
        preprocessor_path = os.getenv("PREPROCESSOR_PATH", "preprocessor.pkl")
        preprocessor_created = joblib.load(preprocessor_path)
        logging.info("Preprocessor loaded successfully.")
        return preprocessor_created
    except Exception as e:
        logging.error(f"Error loading preprocessor: {e}")
        raise RuntimeError("Failed to load the preprocessor.")


class ModelHandler:
    """
    This class is responsible for handling the loading of the trained machine learning model and the preprocessor.
    """

    def __init__(self):
        self.model = load_model()
        self.preprocessor = load_preprocessor()

    def preprocess_input(self, data: dict):
        """
        This method is responsible for converting the input to DataFrame, performing feature engineering features,
        and applying preprocessing to the initial features.
        """
        try:
            input_df = pd.DataFrame([data])
            # Create new features while avoiding division by zero errors
            input_df['rooms_per_house'] = input_df['total_rooms'] / input_df['households'].replace(0, 1)
            input_df['bedroom_ratio'] = input_df['total_bedrooms'] / input_df['total_rooms'].replace(0, 1)
            input_df['people_per_house'] = input_df['population'] / input_df['households'].replace(0, 1)

            # Apply preprocessing and return the preprocessed dataframe
            return self.preprocessor.transform(input_df)

        # Handling the case of invalid input data and log the issue
        except Exception as e:
            logging.error(f"Error during preprocessing: {e}")
            raise ValueError("Invalid input data. Ensure proper values and formats.")


# Create a model handler instance
model_handler = ModelHandler()

# Initialize FastAPI app
app = FastAPI()


# Define an endpoint for making house price predictions
@app.post("/predict/")
async def predict(data: dict):
    """
    This method is responsible for receiving input data, processing it, and returning the predicted house value.
    """
    try:
        # Validate manually using Pydantic (to catch invalid data early)
        validated_data = InputData(**data)
        input_features = model_handler.preprocess_input(validated_data.dict())

        # Calculate the prediction using the trained model
        prediction = model_handler.model.predict(input_features)

        # Return the predicted house value as a JSON response
        return {"predicted_house_value": float(prediction[0])}

    # Handling invalid inputs such as missing or incorrect data types
    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail="Invalid input data format.")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    # Handling unexpected errors during prediction and log the issue
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
