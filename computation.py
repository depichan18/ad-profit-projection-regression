import numpy as np

# --- 1. Case Study Data ---
# Ad Spend (independent variable X) and Profit (dependent variable Y)
campaigns = ["A", "B", "C", "D", "E", "F", "G"]
x = [2, 3, 5, 7, 1, 4, 6]      # Ad Spend (in $1,000s)
y = [65, 70, 75, 85, 60, 72, 78]  # Profit (in $1,000s)

# --- 2. Create the Design Matrix ---
X = np.column_stack((np.ones(len(x)), x))  # Add intercept column
y_vec = np.array(y).reshape(-1, 1)

# --- 3. Compute β using Normal Equation ---
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xt_y = X.T @ y_vec
beta = XtX_inv @ Xt_y  # Coefficients [intercept, slope]

# --- 4. Compute Predicted y (Projection) and Residuals ---
y_hat = X @ beta
residuals = y_vec - y_hat

# --- 5. Pretty Printer ---
def format_number(val):
    if abs(val - round(val)) < 1e-10:
        return str(int(round(val)))
    else:
        return f"{val:.2f}"

def print_matrix(name, matrix, labels=None):
    print(f"\n{name}:")
    for i, row in enumerate(matrix):
        label = f"{labels[i]}: " if labels else ""
        print(f"{label}[" + "  ".join(format_number(val) for val in row) + "]")

# --- 6. Display Results ---
print_matrix("Design Matrix X", X)
print_matrix("Actual y (Profit)", y_vec, campaigns)
print_matrix("Predicted y (Projection ŷ)", y_hat, campaigns)
print_matrix("Residuals (y - ŷ)", residuals, campaigns)
print_matrix("Beta (Coefficients)", beta)

# --- 7. Model Summary ---
intercept = beta[0, 0]
slope = beta[1, 0]
print(f"\nFinal Regression Equation:")
print(f"ŷ = {format_number(intercept)} + {format_number(slope)} × x (Ad Spend in $1,000s)")
print("\nConclusion:")
print(f"Each additional $1,000 of ad spend is predicted to increase the profit by about {format_number(slope)} thousand dollars.")
