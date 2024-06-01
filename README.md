# Optimizing Logistic Regression for Wine Classification

This project uses Logistic Regression to classify types of wine based on various features. The project includes preprocessing steps such as data imputation, outlier removal, and feature scaling. The Logistic Regression model's performance is evaluated using cross-validation and accuracy scores. The project also focuses on improving the efficiency of the model by tuning the 'max_iter' parameter.

## Project Structure

The project has the following structure:

- `main.py`: This is the main Python script where the data is loaded, preprocessed, and the Logistic Regression model is trained and evaluated.
- `utils/`: This directory contains utility scripts used in the main script.
  - `imputer_wine.py`: This script contains the function for imputing missing values in the dataset.
  - `outlier_removal.py`: This script contains the function for removing outliers from the dataset.
- `data/`: This directory contains the dataset used in the project.

## How to Run

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Run `python main.py` to execute the script.

## Dependencies

This project requires Python and the following Python libraries installed:

- NumPy
- pandas
- scikit-learn

If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.