from simple_term_menu import TerminalMenu
import numpy as np
import sys


class PricePrediction:

    def __init__(self, theta: list[float]=np.zeros(2)):
        self.theta = theta
        self.verify_training()

    def verify_training(self) -> None:
        if np.allclose(self.theta, 0, atol=1e-7):
            print("Warning: the model has not been trained yet.")
            options = [
                "Run with theta[0], theta[1] = 0, 0 (the predictions will not be accurate!)",
                "Train the model",
            ]
            menu = TerminalMenu(options)
            index = menu.show()
            if options[index] == "Train the model":
                pass
                # Add logic for calling the training program
            else:
                pass
                # Calculate price with broken values

PricePrediction()