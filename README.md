# Lab Sheet-03: Regression Models (Supervised Learning)


## About this project

This is my work for Lab Sheet-03 (Lab Assessment on Supervised Learning -
Regression Models). It has 35 programs. They build Simple, Multiple and
Polynomial Regression models, check them with MAE, MSE, RMSE and R2, try
feature scaling, use a second dataset, and save the models with Joblib.

## What is in the folder

| File / Folder | What it does |
|---|---|
| `lab_sheet_03_all_programs.py` | All 35 programs in one file. Each program is its own cell. |
| `datasets/advertising.csv` | Main dataset: TV, Radio and Newspaper budget, and Sales. |
| `datasets/diabetes.csv` | Second dataset (from Scikit-learn), saved by Program 33. |
| `models/` | Trained models saved with Joblib (Program 34). |
| `outputs/` | The graphs saved by the programs. |
| `requirements.txt` | The list of libraries to install. |

## Datasets

- `advertising.csv`: 400 rows, inputs are TV, Radio, Newspaper, output is Sales.
  This is a sample dataset made for the lab. You can replace it with the real
  Advertising dataset (same column names) and run the file again.
- Diabetes dataset: 442 patients, 10 features, output is disease progression.

## Libraries used

NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Joblib, Jupyter Notebook.
Python version: 3.11 or above.

## How to run it

1. Open the project folder in VS Code.
2. Open the terminal and make a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

   (On Linux or Mac, use `source venv/bin/activate`.)

3. Install the libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the whole file:

   ```
   python lab_sheet_03_all_programs.py
   ```

   Or open the file in VS Code and click **Run Cell** above any program.
   You can also open the `.ipynb` file and run the cells there.

Program 12 asks you to type a TV budget. Press Enter to use 150.

## What the programs cover

- **Programs 1-12:** Load and explore the data, split it into train and test,
  build a Simple Linear Regression model, plot the line, and predict new values.
- **Programs 13-17:** Multiple Linear Regression and the effect of each input.
- **Programs 18-23:** Polynomial Regression (degree 2 and 3), curves, and
  accuracy for different degrees.
- **Programs 24-30:** MAE, MSE, RMSE, R2, model comparison, and error plots.
- **Programs 31-35:** Feature scaling, a second dataset, and saving and
  loading a model with Joblib.


