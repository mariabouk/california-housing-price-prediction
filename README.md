# California Housing Price Prediction

## Overview
This project is a machine learning-based API designed to predict housing prices in California. It utilizes the "California Housing Prices" ([dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)), with an XGBoost model for regression. The XGBoost Model has been selected as the best between different models (XGBoost, Random Forest, SVM, Linear Regression). The application is built using FastAPI and provides a REST API for serving predictions.

## Features
- Data preprocessing with feature engineering
- Model training and evaluation
- Detection of the best model -> XGBoost
- A modular and scalable software architecture
- REST API implementation using FastAPI
- Error handling and logging for robustness

## Repository Structure
```
├── main.py                                    # FastAPI application with prediction endpoint
├── preprocessing_and_model_creation.ipynb     # Jupyter Notebook for data preprocessing and model training
├── Models/                                    # Directory containing the best trained model (XGBoost)
├── preprocessor.pkl                           # Preprocessor used for data transformation (also stored in the Models file)
├── requirements.txt                           # Dependencies for running the project
└── README.md                                  # Project documentation
```

## Setup Instructions
### Prerequisites
Ensure you have Python installed (>=3.8). Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

### Running the API
1. Ensure the trained model (`CaliforniaBestHousingModel.json`) and `preprocessor.pkl` are present in the appropriate directories.
2. Start the FastAPI server:
```sh
uvicorn main:app --reload
```
3. Access the API documentation at `http://127.0.0.1:8000/docs`

## API Usage
### Endpoint: `/predict/`
- **Method:** `POST`
- **Input:** JSON object with the following fields:
  ```json
  {
    "latitude": -122.23,
    "longitude": 37.38,
    "housing_median_age": 880.0,
    "population": 129.0,
    "total_rooms": 322.0,
    "total_bedrooms": 126.0,
    "households": 1000.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }
  ```
- **Output:**
  ```json
  {
  "predicted_house_value": 331059.0625
  }
  ```

## Software Architecture
- **Data Preprocessing:** The dataset is cleaned, missing values handled, outliers are removed, and additional features (rooms per house, bedroom ratio, people per house) are added.
- **Model Training:** The XGBoost model is trained using the processed dataset and saved for later inference.
- **Prediction Pipeline:** The API receives input, preprocesses it, applies the trained model, and returns a price prediction.
- **Error Handling:** The system catches and logs errors related to data input, preprocessing, and model inference.

## Future Improvements
- Implement a database for storing predictions
- Deploy the API as a cloud service (AWS/GCP/Azure)
- Enhance model performance with hyperparameter tuning and additional features
