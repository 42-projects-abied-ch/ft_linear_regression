import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import subprocess
from simple_term_menu import TerminalMenu
from train import Trainer

# new_mileage = 240000
# normalized_new_mileage = (new_mileage - trainer.mean_km) / trainer.std_km
# estimated_price = trainer.estimate_price(normalized_new_mileage, theta)
# print(f"Estimated Price for {new_mileage} km mileage: {estimated_price:.2f}")

def train(trainer: Trainer):
    learning_rates = np.arange(0.001, 0.15, 0.02).tolist()
    convergence_thresholds = [1e-3, 1e-4, 1e-5, 1e-6]

    trainer.hyperparameter_tuning(
        trainer.mileages, learning_rates, convergence_thresholds
    )

def load_model():
    with open("model.json", "r"):
        subprocess.run(["python3", "calculate_price.py"])

def main():
    os.system("clear")
    trainer = Trainer()
    try:
        load_model()
    except FileNotFoundError:
        print("WARNING: The model has not yet been trained. What would you like to do?")
        options = ["Train the model", "Exit"]
        menu = TerminalMenu(options)
        index = menu.show()
        if options[index] == "Exit":
            sys.exit(0)
        else:
            train(trainer)
            load_model()

if __name__ == "__main__":
    main()
