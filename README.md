# Kriging Module

A reusable Python module for performing Ordinary Kriging, AR1 Cokriging, (based on the work of Forrester) and Hierarchical Kriging (based on the work of Han and Görtz) with uncertainty quantification.. This module uses a power exponential kernel and optimises hyperparameters via differential evolution.

## Features

- **Automatic Hyperparameter Tuning:** Optimises the model’s hyperparameters using differential evolution.
- **Prediction with Uncertainty:** Provides both the mean prediction and an estimate of the prediction uncertainty (RMSE).
- **Reusable Design:** Encapsulated in a class for easy integration into other projects.
- **Numerical Stability:** Includes jitter in the correlation matrix for improved stability.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/thomasongley/kriging-module.git
   cd kriging-module

2. **Install Dependencies:**

    Ensure you have NumPy and SciPy installed. For running the example, you will also needMatplotlib:
    
    ```bash
    pip install numpy scipy matplotlib

## Usage

Import the module and use the KrigingModel class to fit your data and make predictions:

```bash
from kriging import KrigingModel
import numpy as np
import matplotlib.pyplot as plt

# Define the test function (e.g., Forrester function)
def forrester(x):
    return (6 * x - 2)**2 * np.sin(12 * x - 4)

# Generate training data
X_train = np.linspace(0, 1, 5).reshape(-1, 1)
y_train = forrester(X_train)

# Generate test data
X_test = np.linspace(0, 1, 201).reshape(-1, 1)

# Create and fit the kriging model
model = KrigingModel(bounds=[(-3, 2)])
model.fit(X_train, y_train)

# Predict using the fitted model
y_pred, rmse = model.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(X_train, y_train, 'ro', label='Training Data')
plt.plot(X_test, y_pred, 'b-', label='Prediction')
plt.fill_between(X_test.ravel(), y_pred - 2 * rmse, y_pred + 2 * rmse,
                 color='gray', alpha=0.5, label='Confidence Interval')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Kriging Prediction')
plt.show()
```

## Project Structure

```bash

kriging-module/
├── kriging.py         # The main module containing the KrigingModel class
├── README.md          # This file
├── .gitignore         # Files and folders to ignore in git
└── requirements.txt   # (Optional) List of dependencies
```

## Acknowledgments

This kriging module contains a translation and adaptation of MATLAB code from *Engineering Design via Surrogate Modelling* by Forrester *et al.* [^1]; originally provided under the GNU Lesser General Public License (LGPL). All credit for the original work goes to the original author. The Hierarchical Kriging class is an implementation of the method proposed by Han and  Görtz [^2].

[^1]: Alexander I. J. Forrester, András Sóbester, Andy J. Keane, *Engineering Design via Surrogate Modelling*, John Wiley & Sons, 2008. 
[^2]: Zhong-Hua Han, Stefan Görtz, *Hierarchical Kriging Model for Variable-Fidelity Surrogate Modeling*, AIAA Journal, 2012.
