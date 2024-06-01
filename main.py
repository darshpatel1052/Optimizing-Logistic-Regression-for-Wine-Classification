import numpy as np
import pandas as pd
from utils.imputer_wine import impute_and_transform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils.outlier_removal import outlier_removal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

def evaluate_model(x, y):
    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Create a logistic regression model with increased max_iter
    model = LogisticRegression(max_iter=5000)

    # Train the model
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Calculate and return the accuracy of the model
    return accuracy_score(y_test, y_pred)

# Assuming 'type' is your target column and the rest are features
df = pd.read_csv('wine.csv')
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
y = df['type'].copy()  # Make a copy of 'type' column

# Impute and transform the data
df, y = impute_and_transform(df, y)  # Update your impute_and_transform function to also accept and return y

# Accuracy before outlier removal
accuracy = evaluate_model(df.drop('type', axis=1), y)
print('Accuracy after imputation:', accuracy)

# Remove outliers
x, y = outlier_removal(df.drop('type', axis=1), y)  # Update your outlier_removal function to also accept and return y

# Accuracy after outlier removal
accuracy = evaluate_model(x, y)
print('Accuracy after outlier removal:', accuracy)

# Scale the data
scalar = StandardScaler()
x = scalar.fit_transform(x)
x_df = pd.DataFrame(x, columns=df.columns[1:])

cov_matrix = np.cov(x_df.T)
# Compute the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Make a list of (eigenvalue, eigenvector) tuples
eigenpairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigenpairs.sort(key=lambda x: x[0], reverse=True)

# Choose the top k eigenvectors
k = 9
matrix_w = np.hstack([eigenpairs[i][1].reshape(len(eigenvalues), 1) for i in range(k)])



# Transform the original dataset to the new feature subspace
x_pca = x_df.dot(matrix_w)
accuracy = evaluate_model(x_pca, y)
print('Accuracy after dimensionality reduction:', accuracy)








