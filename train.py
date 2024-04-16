import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mileages = pd.read_csv("data.csv")

mean_km = mileages["km"].mean()
std_km = mileages["km"].std()

mileages["normalized_km"] = (mileages["km"] - mean_km) / std_km

learning_rate = 0.05
iterations = 500
convergence_threshold = 1e-4

def estimate_price(normalized_mileage: float, theta: list[float]) -> float:
    return theta[0] + (theta[1] * normalized_mileage)

def gradient_descent(mileages, learning_rate, convergence_threshold, iterations=500):

    theta = np.zeros(2)

    mean_km = mileages["km"].mean()
    std_km = mileages["km"].std()
    mileages["normalized_km"] = (mileages["km"] - mean_km) / std_km

    for i in range(iterations):
        predictions = estimate_price(mileages["normalized_km"], theta)
        errors = predictions - mileages["price"]

        gradient_theta_0 = np.mean(errors)
        gradient_theta_1 = np.mean(errors * mileages["normalized_km"])

        new_theta_0 = theta[0] - learning_rate * gradient_theta_0
        new_theta_1 = theta[1] - learning_rate * gradient_theta_1

        if np.abs(new_theta_0 - theta[0]) < convergence_threshold and np.abs(new_theta_1 - theta[1]) < convergence_threshold:
            return i + 1, theta

        theta[0], theta[1] = new_theta_0, new_theta_1

    return iterations, theta 

def hyperparameter_tuning(mileages: pd.DataFrame, learning_rates: list[float], convergence_thresholds: list[float]) -> tuple[float, float, int, np.ndarray]:
    best_learning_rate = None
    best_convergence_threshold = None
    best_iterations = float("inf")
    best_theta = None

    for lr in learning_rates:
        for ct in convergence_thresholds:
            iters, theta = gradient_descent(mileages, lr, ct)
            if iters < best_iterations:
                best_iterations = iters
                best_learning_rate = lr
                best_convergence_threshold = ct
                best_theta = theta

    return best_learning_rate, best_convergence_threshold, best_iterations, best_theta

learning_rates = np.arange(0.001, 0.15, 0.02).tolist()
convergence_thresholds = [1e-3, 1e-4, 1e-5, 1e-6]

best_lr, best_ct, best_iters, theta = hyperparameter_tuning(mileages, learning_rates, convergence_thresholds)
print(f"Best Learning Rate: {best_lr}, Best Convergence Threshold: {best_ct}, Iterations: {best_iters}, Theta: {theta}")

x_values = np.linspace(mileages["km"].min(), mileages["km"].max(), 400)
x_normalized = (x_values - mean_km) / std_km 
y_values = estimate_price(x_normalized, theta)

plt.figure(figsize=(10, 6))
plt.scatter(mileages["km"], mileages["price"], color='blue', label='Actual Prices')
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.title('Car Price vs. Mileage')
plt.xlabel('Mileage (km)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.savefig("plot.png")
plt.show()

new_mileage = 240000
normalized_new_mileage = (new_mileage - mean_km) / std_km
estimated_price = estimate_price(normalized_new_mileage, theta)
print(f"Estimated Price for {new_mileage} km mileage: {estimated_price:.2f}")
