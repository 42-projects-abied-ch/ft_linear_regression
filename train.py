import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mileages = pd.read_csv("data.csv")

mean_km = mileages["km"].mean()
std_km = mileages["km"].std()

mileages["normalized_km"] = (mileages["km"] - mean_km) / std_km

theta = np.zeros(2)
learning_rate = 0.05
iterations = 500
convergence_threshold = 1e-6

def estimate_price(normalized_mileage):
    return theta[0] + (theta[1] * normalized_mileage)

for i in range(iterations):
    predictions = estimate_price(mileages["normalized_km"])
    errors = predictions - mileages["price"]

    gradient_theta_0 = np.mean(errors)
    gradient_theta_1 = np.mean(errors * mileages["normalized_km"])

    new_theta_0 = theta[0] - learning_rate * gradient_theta_0
    new_theta_1 = theta[1] - learning_rate * gradient_theta_1

    if np.abs(new_theta_0 - theta[0]) < convergence_threshold and np.abs(new_theta_1 - theta[1]) < convergence_threshold:
        print(f"Convergence threshold reached at iteration {i + 1}")
        break

    theta[0], theta[1] = new_theta_0, new_theta_1

    if i % 100 == 0 or i == iterations - 1:
        print(f"Iteration {i + 1}: theta[0] = {theta[0]}, theta[1] = {theta[1]}")

x_values = np.linspace(mileages["km"].min(), mileages["km"].max(), 400)
x_normalized = (x_values - mean_km) / std_km 
y_values = estimate_price(x_normalized)

plt.figure(figsize=(10, 6))
plt.scatter(mileages["km"], mileages["price"], color='blue', label='Actual Prices')
plt.plot(x_values, y_values, color='red', label='Regression Line')
plt.title('Car Price vs. Mileage')
plt.xlabel('Mileage (km)')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

new_mileage = 240000
normalized_new_mileage = (new_mileage - mean_km) / std_km
estimated_price = estimate_price(normalized_new_mileage)
print(f"Estimated Price for {new_mileage} km mileage: {estimated_price:.2f}")
