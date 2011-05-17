import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .utils import remove_outliers_iqr

def main():
    df = remove_outliers_iqr( pd.read_csv('listings_filtered.csv'), 'price' )

    df = pd.get_dummies(df, columns=[
        'neighbourhood_cleansed', 'property_type'
    ], drop_first=True)
    
    feature_cols = ['review_scores_rating'] + [col for col in df.columns if col.startswith('neighbourhood_cleansed_') or col.startswith('property_type_')]
    X = df[feature_cols]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    print(f"R^2 Score: {r2:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("actual price")
    plt.ylabel("predicted price")
    plt.title("price prediction: neighborhood & property type")
    plt.tight_layout()
    plt.show()
