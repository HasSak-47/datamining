import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def main():
    df = pd.read_csv('listings_filtered.csv')
    
    # 1. Histogram
    for col in ["number_of_reviews", "review_scores_rating"]:
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogram of {col}")
        plt.show()
    
    # 2. Boxplot
    for col in ["price", "number_of_reviews", "review_scores_rating"]:
        cleaned_df = remove_outliers_iqr(df, col)
        sns.boxplot(y=cleaned_df[col])
        plt.title(f"Boxplot of {col} (Outliers Removed)")
        plt.show()
    
    # 3. Pie Chart (room type distribution)
    df["room_type"].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title("Room Type Distribution")
    plt.show()
    
    # 4. Scatter Plot
    sns.scatterplot(x="number_of_reviews", y="price", data=df)
    plt.title("Price vs Reviews")
    plt.show()
