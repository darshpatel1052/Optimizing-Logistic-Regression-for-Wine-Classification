import pandas as pd
def outlier_removal(df, y):
    cols = ['volatile acidity', 'free sulfur dioxide', 'density','total sulfur dioxide','fixed acidity','citric acid','chlorides','sulphates']

    mask = pd.Series(True, index=df.index)
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = mask & (df[col] >= lower_bound) & (df[col] <= upper_bound)

    return df[mask], y[mask]
