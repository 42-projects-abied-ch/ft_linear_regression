from simple_term_menu import TerminalMenu
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os


class PricePrediction:

    def __init__(self):
        self.load_model()
        self.do_visualize = False

    def estimate_price(self, normalized_mileage: float) -> float:
        return self.theta[0] + (self.theta[1] * normalized_mileage)

    def normalize_mileage(self, mileage: float) -> float:
        return (mileage - self.mean_km) / self.std_km

    def load_model(self) -> None:
        model = {}
        with open("model.json", "r") as model:
            model = json.load(model)
        self.theta = [model["theta_0"], model["theta_1"]]
        self.mean_km = model["mean_km"]
        self.std_km = model["std_km"]

    def prompt_user(self) -> None:
        os.system("clear")
        mileage = float(input("Please enter the mileage of your car: "))
        normalized_mileage = self.normalize_mileage(mileage)
        estimated_price = self.estimate_price(normalized_mileage)
        os.system("clear")
        print(f"The estimated price for a mileage of {mileage} is: {estimated_price}€\n")
        print("Would you like to visualize the predicted datapoint in relation to the training data?")
        options = [
            "Yes",
            "No"]
        menu = TerminalMenu(options)
        index = menu.show()
        if options[index] == "Yes":
            self.visualize({"km": mileage, "price": estimated_price})

    def visualize(self, estimated_datapoint: dict) -> None:
        mileages = pd.read_csv("data.csv")
        x_values = np.linspace(
            mileages["km"].min(), mileages["km"].max(), 400
        )
        x_normalized = (x_values - self.mean_km) / self.std_km
        y_values = self.estimate_price(x_normalized)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            mileages["km"],
            mileages["price"],
            color="blue",
            label="Actual Prices",
        )
        plt.plot(x_values, y_values, color="red", label="Regression Line")

        plt.scatter(
        [estimated_datapoint["km"]],
        [estimated_datapoint["price"]],
        color="green",
        label="Estimated Price",
        marker='o',  # You can change the marker style if you want
        s=100  # Adjusts the size of the point
    )

        plt.title("Car Price vs. Mileage")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price (€)")
        plt.legend()
        plt.grid(True)
        plt.savefig("plot.png")
        plt.show()


PricePrediction().prompt_user()

