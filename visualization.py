import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = [2, 3, 5, 7, 1, 4, 6]
y = [65, 70, 75, 85, 60, 72, 78]

# Linear regression coefficients
X = np.column_stack((np.ones(len(x)), x))
y_vec = np.array(y).reshape(-1, 1)
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xt_y = X.T @ y_vec
beta = XtX_inv @ Xt_y

y_hat = X @ beta

# Flatten vectors for plotting
x = np.array(x)
y_hat_flat = y_hat.flatten()

# Plot
plt.scatter(x, y, label='Actual Profit (y)', color='blue')
plt.plot(x, y_hat_flat, label='Predicted Profit (ŷ)', color='red')
plt.xlabel("Ad Spend (in $1,000s)")
plt.ylabel("Profit (in $1,000s)")
plt.title("Linear Regression: Predicting Profit from Advertising Spend")
plt.grid(True)
plt.legend()
plt.show()
