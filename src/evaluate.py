import numpy as np
import pandas as pd
import json

def estimate_price(normalized_mileage: float, theta: list[float]) -> float:
    return theta[0] + (theta[1] * normalized_mileage)


def normalize_mileage(mileage: float, mean_km: float, std_km: float) -> float:
    return (mileage - mean_km) / std_km


def calculate_evaluation_metrics(mileages: pd.DataFrame) -> dict:
    """
    Calculates commonly used metrics for linear regression performance evaluation.

        - Mean Squared Error (MSE): The average of the squared differences between
          the predicted and actual values.

        - Root Mean Squared Error (RMSE): The square root of the Mean Squared Error.
          This metric is more commonly used, because it is in the same unit as the target variable.

        - Mean Absolute Error (MAE): The average of the absolute differences between the predicted and actual values.

        - R-Squared value: The proportion of the variance in the dependent variable that
          is predictable from the independent variable.
    """
    mse = np.mean((mileages["price"] - mileages["price_prediction"]) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(mileages["price"] - mileages["price_prediction"]))
    ss_res = np.sum((mileages["price"] - mileages["price_prediction"]) ** 2)
    ss_tot = np.sum((mileages["price"] - np.mean(mileages["price"])) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R-Squared": r_squared,
    }


def evaluate(mileages: pd.DataFrame) -> dict:
    with open("model.json", "r") as model_file:
        model = json.load(model_file)
    theta = [model["theta_0"], model["theta_1"]]
    mean_km = model["mean_km"]
    std_km = model["std_km"]
    mileages = pd.read_csv("data.csv")
    mileages["normalized_mileage"] = normalize_mileage(mileages["km"], mean_km, std_km)
    mileages["price_prediction"] = estimate_price(mileages["normalized_mileage"], theta)
    metrics = calculate_evaluation_metrics(mileages)
    return metrics