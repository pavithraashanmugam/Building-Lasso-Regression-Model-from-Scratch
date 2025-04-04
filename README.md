# Lasso Regression Model Implementation

This repository provides an implementation of **Lasso Regression** from scratch in Python. The model is designed to predict the target variable \( Y \) based on input features \( X \) using **Lasso regularization**. The Lasso method is particularly useful when dealing with high-dimensional datasets where feature selection is important. It adds a penalty term to the linear regression model to shrink some of the coefficients to zero, effectively performing feature selection.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Usage](#usage)
4. [Methods](#methods)
5. [Dependencies](#dependencies)
6. [Example](#example)

## Overview

Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a linear regression model that incorporates a penalty (regularization) term based on the absolute values of the coefficients. The penalty term is controlled by a hyperparameter, **lambda**. The objective of the Lasso Regression model is to minimize the residual sum of squares (RSS) along with the penalty term for the coefficients.

The main idea behind Lasso is to apply **L1 regularization**, which forces some of the coefficients to be exactly zero, leading to a sparse solution and potentially improving model interpretability.

## How It Works

The Lasso Regression model tries to minimize the following cost function:

\[
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2 + \lambda \sum_{j=1}^{n} |w_j|
\]

Where:
- \( y^{(i)} \) is the true target value for the \(i\)-th sample.
- \( \hat{y}^{(i)} \) is the predicted target value for the \(i\)-th sample.
- \( w_j \) represents the weights (coefficients) of the model.
- \( \lambda \) is the regularization parameter that controls the amount of shrinkage applied to the weights.
- \( m \) is the number of training examples.
- \( n \) is the number of features.

The class `Lasso_Regression` includes methods for fitting the model to training data and making predictions.

## Usage

To use this model, follow the steps below:

1. **Create a `Lasso_Regression` object** and specify the learning rate, the number of iterations, and the regularization parameter lambda.
2. **Fit the model** to your dataset using the `fit()` method.
3. **Make predictions** using the `predict()` method.

## Methods

### `__init__(self, learning_rate, no_of_iterations, lambda_parameter)`

Initializes the hyperparameters for the model:
- `learning_rate`: The rate at which the model's weights are updated during gradient descent.
- `no_of_iterations`: The number of iterations for updating weights.
- `lambda_parameter`: The regularization parameter that controls the amount of shrinkage for the weights.

### `fit(self, X, Y)`

Fits the model to the data by updating the weights and bias using gradient descent:
- `X`: The input features, where each row represents a data point and each column represents a feature.
- `Y`: The target variable values.

### `update_weights(self)`

Updates the weights and bias using the gradient descent algorithm. The gradient of the loss function with respect to the weights and bias is computed, and the weights are updated accordingly.

### `predict(self, X)`

Generates predictions using the learned weights and bias:
- `X`: The input features for which predictions are to be made.

## Dependencies

This implementation only depends on **NumPy** for numerical calculations. There are no built-in machine learning libraries used.

- **NumPy**: A package for numerical computing in Python, used for matrix operations and efficient calculations.

To install the required dependency, you can use:

```bash
pip install numpy
```

## Example

Here is an example of how to use the `Lasso_Regression` model:

```python
import numpy as np

# Example dataset (X: features, Y: target)
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
Y = np.array([1, 2, 3, 4, 5])

# Initialize the Lasso Regression model
model = Lasso_Regression(learning_rate=0.01, no_of_iterations=1000, lambda_parameter=0.1)

# Fit the model to the data
model.fit(X, Y)

# Make predictions on the dataset
predictions = model.predict(X)

print("Predictions:", predictions)
```

This will output the predicted values for the given input features \( X \).

---

## Conclusion

This implementation of Lasso Regression helps you understand how regularization works and how the Lasso method can perform feature selection. It provides an intuitive, step-by-step approach to applying Lasso Regression without relying on built-in machine learning libraries.
