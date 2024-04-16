import pandas as pd
import numpy as np
import json


class Trainer:

    def __init__(self):
        self.mileages = pd.read_csv("data.csv")
        self.mean_km = self.mileages["km"].mean()
        self.std_km = self.mileages["km"].std()
        self.mileages["normalized_km"] = self.normalize_mileage(self.mileages["km"])
        self.best_learning_rate = None
        self.best_convergence_threshold = None
        self.best_iterations = float("inf")
        self.best_theta = None

    def normalize_mileage(self, mileage: int) -> float:
        return (mileage - self.mean_km) / self.std_km

    def estimate_price(self, normalized_mileage: float, theta: list[float]) -> float:
        return theta[0] + (theta[1] * normalized_mileage)

    def convergence_threshold_reached(
        self, new_theta: list[float], theta: list[float], convergence_threshold: float
    ) -> bool:
        return (
            np.abs(new_theta[0] - theta[0]) < convergence_threshold
            and np.abs(new_theta[1] - theta[1]) < convergence_threshold
        )

    def calculate_gradient(self, errors: list[float]) -> list[float]:
        gradient_theta = [0] * 2
        gradient_theta[0] = np.mean(errors)
        gradient_theta[1] = np.mean(errors * self.mileages["normalized_km"])
        return gradient_theta

    def calculate_new_theta(
        self, theta: list[float], gradient_theta: list[float], learning_rate: float
    ) -> list[float]:
        new_theta = [0, 0]
        new_theta[0] = theta[0] - learning_rate * gradient_theta[0]
        new_theta[1] = theta[1] - learning_rate * gradient_theta[1]
        return new_theta

    def model_to_json(self) -> None:
        model_params = {
            "theta_0": self.best_theta[0],
            "theta_1": self.best_theta[1],
            "mean_km": self.mean_km,
            "std_km": self.std_km,
        }
        with open("model.json", "w") as model:
            json.dump(model_params, model, indent=4)

    def gradient_descent(
        self,
        mileages: pd.DataFrame,
        learning_rate: float,
        convergence: float,
        iterations: int = 500,
    ) -> tuple[int, list[float]]:

        theta = np.zeros(2)

        for i in range(iterations):
            predictions = self.estimate_price(self.mileages["normalized_km"], theta)
            errors = predictions - mileages["price"]

            grad_t = self.calculate_gradient(errors)
            new_t = self.calculate_new_theta(theta, grad_t, learning_rate)

            if self.convergence_threshold_reached(new_t, theta, convergence) == True:
                return i + 1, theta

            theta = new_t

        return iterations, theta

    def update_hyperparameters(
        self,
        iters: int,
        learning_rate: float,
        convergence: float,
        theta: list[float],
    ) -> None:
        self.best_iterations = iters
        self.best_learning_rate = learning_rate
        self.best_convergence_threshold = convergence
        self.best_theta = theta

    def hyperparameter_tuning(
        self,
        mileages: pd.DataFrame,
        learning_rates: list[float],
        convergence_thresholds: list[float],
    ) -> tuple[float, float, int, np.ndarray]:

        for lr in learning_rates:
            for ct in convergence_thresholds:
                iters, theta = self.gradient_descent(mileages, lr, ct)
                if iters < self.best_iterations:
                    self.update_hyperparameters(iters, lr, ct, theta)

        self.model_to_json()

        return self.best_theta
