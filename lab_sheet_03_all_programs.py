# %% [markdown]
# # Lab Sheet-03: Supervised Learning (Regression Models)
# **MCA III Semester (Session 2026-2027) - COER University, Roorkee**
#
# Each `# %%` block is one experiment (a separate cell in VS Code / Jupyter).
#
# Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib
# Datasets: `datasets/advertising.csv` (main), Diabetes dataset from Scikit-learn (Program 33)

# %% [markdown]
# ## Setup: imports, paths and helper functions

# %%
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()
    if BASE_DIR.name == "notebooks":
        BASE_DIR = BASE_DIR.parent

DATA_PATH = BASE_DIR / "datasets" / "advertising.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_dataset(path=DATA_PATH):
    """Load the CSV file with basic validation and exception handling."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        data = pd.read_csv(path)
        if data.empty:
            raise ValueError("Dataset is empty")
        return data
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as err:
        print("Error while loading dataset:", err)
        raise


def evaluate(y_true, y_pred):
    """Return MAE, MSE, RMSE and R2 in a dictionary."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


# %% [markdown]
# # Part A: Data Loading and Simple Linear Regression (Programs 1-12)

# %% [markdown]
# ## Program 1: Load a regression dataset using Pandas

# %%
df = load_dataset()
print("Dataset loaded. Shape:", df.shape)

# %% [markdown]
# ## Program 2: First and last five records

# %%
print("First 5 records:")
print(df.head())
print("\nLast 5 records:")
print(df.tail())

# %% [markdown]
# ## Program 3: Dataset information and descriptive statistics

# %%
df.info()
print()
print(df.describe().round(2))
print("\nMissing values:\n", df.isnull().sum())

# %% [markdown]
# ## Program 4: Identify input (independent) and output (dependent) variables

# %%
feature_columns = ["TV", "Radio", "Newspaper"]  # independent variables (inputs)
target_column = "Sales"                          # dependent variable (output)
X = df[feature_columns]
y = df[target_column]
print("Input variables :", feature_columns)
print("Output variable :", target_column)
print("X shape:", X.shape, "| y shape:", y.shape)

# %% [markdown]
# ## Program 5: Split the dataset into training and testing sets

# %%
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("Training set:", X_train.shape, y_train.shape)
print("Testing set :", X_test.shape, y_test.shape)

# %% [markdown]
# ## Program 6: Implement a Simple Linear Regression model

# %%
# Simple linear regression uses ONE input variable (TV) to predict Sales
X_train_tv = X_train[["TV"]]
X_test_tv = X_test[["TV"]]
simple_model = LinearRegression()
print("Model created:", simple_model)

# %% [markdown]
# ## Program 7: Train the Linear Regression model

# %%
simple_model.fit(X_train_tv, y_train)
print("Simple Linear Regression model trained on", len(X_train_tv), "records")

# %% [markdown]
# ## Program 8: Predict output values using the trained model

# %%
y_pred_simple = simple_model.predict(X_test_tv)
print("First 10 predictions:", y_pred_simple[:10].round(2))

# %% [markdown]
# ## Program 9: Visualize the Linear Regression line

# %%
plt.figure(figsize=(8, 5))
plt.scatter(X_test_tv, y_test, color="tab:blue", alpha=0.7, label="Actual (test data)")
x_line = np.linspace(X["TV"].min(), X["TV"].max(), 100).reshape(-1, 1)
plt.plot(x_line, simple_model.predict(pd.DataFrame(x_line, columns=["TV"])),
         color="red", linewidth=2, label="Regression line")
plt.xlabel("TV advertising budget")
plt.ylabel("Sales")
plt.title("Simple Linear Regression: TV vs Sales")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p09_linear_regression_line.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 10: Compare actual and predicted values

# %%
comparison_simple = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_simple.round(2),
})
comparison_simple["Difference"] = (comparison_simple["Actual"] - comparison_simple["Predicted"]).round(2)
print(comparison_simple.head(10))

# %% [markdown]
# ## Program 11: Regression coefficient and intercept

# %%
print("Coefficient (slope) :", round(simple_model.coef_[0], 4))
print("Intercept           :", round(simple_model.intercept_, 4))
print(f"Equation: Sales = {simple_model.intercept_:.3f} + {simple_model.coef_[0]:.4f} x TV")

# %% [markdown]
# ## Program 12: Predict the output for new user-defined input values

# %%
try:
    try:
        user_text = input("Enter TV budget (press Enter for 150): ")
    except EOFError:  # no keyboard input available (e.g. running as a script file)
        user_text = ""
    tv_budget = float(user_text or 150)
    if tv_budget < 0:
        raise ValueError("Budget cannot be negative")
    new_input = pd.DataFrame({"TV": [tv_budget]})
    predicted_sales = simple_model.predict(new_input)[0]
    print(f"Predicted Sales for TV budget {tv_budget}: {predicted_sales:.2f}")
except ValueError as err:
    print("Invalid input:", err)

# %% [markdown]
# # Part B: Multiple Linear Regression (Programs 13-17)

# %% [markdown]
# ## Program 13: Implement Multiple Linear Regression

# %%
# Multiple linear regression uses ALL input variables: TV, Radio, Newspaper
multi_model = LinearRegression()
print("Model created:", multi_model)
print("Input variables used:", list(X_train.columns))

# %% [markdown]
# ## Program 14: Train the Multiple Linear Regression model

# %%
multi_model.fit(X_train, y_train)
print("Multiple Linear Regression model trained")
print("Intercept:", round(multi_model.intercept_, 4))

# %% [markdown]
# ## Program 15: Predict output values using the testing dataset

# %%
y_pred_multi = multi_model.predict(X_test)
comparison_multi = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_multi.round(2),
})
print(comparison_multi.head(10))

# %% [markdown]
# ## Program 16: Compare actual and predicted values graphically

# %%
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_multi, alpha=0.7, color="tab:green")
limits = [y_test.min(), y_test.max()]
plt.plot(limits, limits, "r--", label="Perfect prediction")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p16_multi_actual_vs_predicted.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 17: Effect of each independent variable on prediction

# %%
coefficients = pd.Series(multi_model.coef_, index=feature_columns)
print("Coefficients (change in Sales for 1 unit increase in the input):")
print(coefficients.round(4))

plt.figure(figsize=(6, 4))
sns.barplot(x=coefficients.index, y=coefficients.values)
plt.title("Effect of each variable on Sales")
plt.ylabel("Coefficient")
plt.savefig(OUT_DIR / "p17_variable_effect.png", dpi=120, bbox_inches="tight")
plt.show()

# Standardized effect: which variable matters most when all are on the same scale
std_effect = pd.Series(
    multi_model.coef_ * X_train.std().values, index=feature_columns
).round(3)
print("\nEffect of a 1 standard-deviation increase:")
print(std_effect)

# %% [markdown]
# # Part C: Polynomial Regression (Programs 18-23)

# %% [markdown]
# ## Program 18: Polynomial Regression of Degree 2

# %%
poly2_model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
poly2_model.fit(X_train_tv, y_train)
y_pred_poly2 = poly2_model.predict(X_test_tv)
print("Degree 2 model trained. R2 on test data:", round(r2_score(y_test, y_pred_poly2), 4))

# %% [markdown]
# ## Program 19: Polynomial Regression of Degree 3

# %%
poly3_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), LinearRegression())
poly3_model.fit(X_train_tv, y_train)
y_pred_poly3 = poly3_model.predict(X_test_tv)
print("Degree 3 model trained. R2 on test data:", round(r2_score(y_test, y_pred_poly3), 4))

# %% [markdown]
# ## Program 20: Compare Linear and Polynomial Regression models

# %%
r2_table = pd.Series({
    "Linear (degree 1)": r2_score(y_test, y_pred_simple),
    "Polynomial (degree 2)": r2_score(y_test, y_pred_poly2),
    "Polynomial (degree 3)": r2_score(y_test, y_pred_poly3),
}).round(4)
print("R2 score on test data (using TV only):")
print(r2_table)

# %% [markdown]
# ## Program 21: Visualize Polynomial Regression curves

# %%
x_grid = pd.DataFrame({"TV": np.linspace(X["TV"].min(), X["TV"].max(), 200)})
plt.figure(figsize=(9, 5))
plt.scatter(X_test_tv, y_test, alpha=0.6, label="Actual (test data)")
plt.plot(x_grid, simple_model.predict(x_grid), label="Linear", color="red")
plt.plot(x_grid, poly2_model.predict(x_grid), label="Degree 2", color="green")
plt.plot(x_grid, poly3_model.predict(x_grid), label="Degree 3", color="purple")
plt.xlabel("TV advertising budget")
plt.ylabel("Sales")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p21_polynomial_curves.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 22: Predict output values using the Polynomial Regression model

# %%
new_tv = pd.DataFrame({"TV": [50, 150, 250]})
print("Predictions for new TV budgets:")
print(pd.DataFrame({
    "TV": new_tv["TV"],
    "Linear": simple_model.predict(new_tv).round(2),
    "Degree 2": poly2_model.predict(new_tv).round(2),
    "Degree 3": poly3_model.predict(new_tv).round(2),
}))

# %% [markdown]
# ## Program 23: Compare prediction accuracy for different polynomial degrees

# %%
degree_results = []
for degree in range(1, 7):
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(X_train_tv, y_train)
    degree_results.append({
        "Degree": degree,
        "Train R2": r2_score(y_train, model.predict(X_train_tv)),
        "Test R2": r2_score(y_test, model.predict(X_test_tv)),
    })
degree_df = pd.DataFrame(degree_results).round(4)
print(degree_df)

plt.figure(figsize=(7, 4))
plt.plot(degree_df["Degree"], degree_df["Train R2"], marker="o", label="Train R2")
plt.plot(degree_df["Degree"], degree_df["Test R2"], marker="o", label="Test R2")
plt.xlabel("Polynomial degree")
plt.ylabel("R2 score")
plt.title("Accuracy for different polynomial degrees")
plt.legend()
plt.grid(True)
plt.savefig(OUT_DIR / "p23_degree_comparison.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Part D: Regression Model Evaluation (Programs 24-30)

# %% [markdown]
# ## Program 24: Mean Absolute Error (MAE)

# %%
# MAE = average of |actual - predicted|
mae_value = mean_absolute_error(y_test, y_pred_simple)
print("MAE (Simple Linear Regression):", round(mae_value, 4))
print("Manual check:", round(np.mean(np.abs(y_test - y_pred_simple)), 4))

# %% [markdown]
# ## Program 25: Mean Squared Error (MSE)

# %%
# MSE = average of (actual - predicted)^2
mse_value = mean_squared_error(y_test, y_pred_simple)
print("MSE (Simple Linear Regression):", round(mse_value, 4))
print("Manual check:", round(np.mean((y_test - y_pred_simple) ** 2), 4))

# %% [markdown]
# ## Program 26: Root Mean Squared Error (RMSE)

# %%
# RMSE = square root of MSE (same unit as the output variable)
rmse_value = np.sqrt(mse_value)
print("RMSE (Simple Linear Regression):", round(rmse_value, 4))

# %% [markdown]
# ## Program 27: R-squared (R2) Score

# %%
r2_value = r2_score(y_test, y_pred_simple)
print("R2 Score (Simple Linear Regression):", round(r2_value, 4))

# %% [markdown]
# ## Program 28: Compare Linear and Polynomial Regression using evaluation metrics

# %%
metrics_table = pd.DataFrame({
    "Simple Linear (TV)": evaluate(y_test, y_pred_simple),
    "Polynomial deg 2 (TV)": evaluate(y_test, y_pred_poly2),
    "Polynomial deg 3 (TV)": evaluate(y_test, y_pred_poly3),
    "Multiple Linear (all)": evaluate(y_test, y_pred_multi),
}).T.round(4)
print(metrics_table)
best_model = metrics_table["R2"].idxmax()
print("\nBest model by R2:", best_model)

# %% [markdown]
# ## Program 29: Interpret the meaning of MSE and R2 values

# %%
best_metrics = metrics_table.loc[best_model]
print(f"Best model: {best_model}")
print(f"- MSE  = {best_metrics['MSE']:.3f}: the average squared gap between actual and predicted Sales.")
print("        Lower is better. 0 means perfect predictions.")
print(f"- RMSE = {best_metrics['RMSE']:.3f}: the typical prediction error, in the same unit as Sales.")
print(f"- R2   = {best_metrics['R2']:.3f}: the model explains about {best_metrics['R2'] * 100:.1f}% "
      "of the variation in Sales.")
print("        1 is perfect, 0 means no better than predicting the mean.")

# %% [markdown]
# ## Program 30: Visualize prediction errors using scatter plots

# %%
residuals = y_test - y_pred_multi
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(y_pred_multi, residuals, alpha=0.7, color="tab:orange")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_xlabel("Predicted Sales")
axes[0].set_ylabel("Error (Actual - Predicted)")
axes[0].set_title("Prediction errors (Multiple Linear Regression)")
sns.histplot(residuals, bins=15, kde=True, ax=axes[1])
axes[1].set_title("Distribution of errors")
plt.tight_layout()
plt.savefig(OUT_DIR / "p30_prediction_errors.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# # Part E: Model Improvement and Analysis (Programs 31-35)

# %% [markdown]
# ## Program 31: Train the regression model using standardized features

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learn mean/std from training data only
X_test_scaled = scaler.transform(X_test)

scaled_model = LinearRegression()
scaled_model.fit(X_train_scaled, y_train)
y_pred_scaled = scaled_model.predict(X_test_scaled)
print("Model trained on standardized features")
print("Coefficients:", dict(zip(feature_columns, scaled_model.coef_.round(4))))

# %% [markdown]
# ## Program 32: Compare performance before and after feature scaling

# %%
scaling_table = pd.DataFrame({
    "Before scaling": evaluate(y_test, y_pred_multi),
    "After scaling": evaluate(y_test, y_pred_scaled),
}).T.round(4)
print(scaling_table)
print("\nNote: for ordinary Linear Regression the predictions stay the same after")
print("scaling. Scaling helps to compare coefficients, and it matters for other")
print("models (like KNN, SVM, or regularized regression).")

# %% [markdown]
# ## Program 33: Regression using another real-world dataset

# %%
# Diabetes dataset from Scikit-learn: predict disease progression after one year
diabetes = load_diabetes(as_frame=True)
diabetes_df = diabetes.frame
diabetes_df.to_csv(BASE_DIR / "datasets" / "diabetes.csv", index=False)
print("Shape:", diabetes_df.shape)
print(diabetes_df.head())

X_d = diabetes_df.drop(columns=["target"])
y_d = diabetes_df["target"]
X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.2, random_state=RANDOM_STATE
)
diabetes_model = LinearRegression().fit(X_d_train, y_d_train)
y_d_pred = diabetes_model.predict(X_d_test)

print("\nPerformance on the Diabetes dataset:")
print(pd.Series(evaluate(y_d_test, y_d_pred)).round(4))

plt.figure(figsize=(6, 5))
plt.scatter(y_d_test, y_d_pred, alpha=0.7)
plt.plot([y_d_test.min(), y_d_test.max()], [y_d_test.min(), y_d_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Diabetes dataset: Actual vs Predicted")
plt.savefig(OUT_DIR / "p33_diabetes_regression.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Program 34: Save the trained regression model using Joblib

# %%
try:
    joblib.dump(simple_model, MODEL_DIR / "simple_linear_model.joblib")
    joblib.dump(multi_model, MODEL_DIR / "multiple_linear_model.joblib")
    joblib.dump(poly2_model, MODEL_DIR / "polynomial_deg2_model.joblib")
    joblib.dump(diabetes_model, MODEL_DIR / "diabetes_model.joblib")
    print("Models saved in the 'models' folder:")
    for file in sorted(MODEL_DIR.glob("*.joblib")):
        print(" -", file.name)
except OSError as err:
    print("Could not save model:", err)

# %% [markdown]
# ## Program 35: Load the saved model and predict new data

# %%
try:
    loaded_model = joblib.load(MODEL_DIR / "multiple_linear_model.joblib")
    new_data = pd.DataFrame({
        "TV": [100, 200],
        "Radio": [20, 35],
        "Newspaper": [30, 60],
    })
    new_predictions = loaded_model.predict(new_data)
    result = new_data.copy()
    result["Predicted Sales"] = new_predictions.round(2)
    print("Predictions from the loaded model:")
    print(result)
except FileNotFoundError:
    print("Model file not found. Run Program 34 first.")

# %% [markdown]
# ## Conclusion
# All 35 experiments of Lab Sheet-03 were completed: Simple, Multiple and
# Polynomial Regression models were trained, evaluated with MAE, MSE, RMSE and
# R2, improved with feature scaling, tested on a second dataset, and saved
# with Joblib.
