import numpy as np
def impute_and_transform(df, y):
    # Impute missing values with the median and apply a log transformation
    for column in ['fixed acidity', 'volatile acidity']:
        df[column].fillna(df[column].median(), inplace=True)
        df[column] = np.log1p(df[column])

    # Impute missing values in 'pH' with the median
    df['pH'] = df['pH'].fillna(df['pH'].median())

    # Find rows with null values in specified columns
    null_rows = df[['sulphates', 'chlorides', 'residual sugar', 'citric acid']].isnull().any(axis=1)

    # Drop these rows from df and y
    df = df[~null_rows]
    y = y[~null_rows]

    return df, y




