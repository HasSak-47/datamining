import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    df = pd.read_csv("listings_filtered.csv")
    

    columns_to_describe = [
        "price", "number_of_reviews", "review_scores_rating",
        "accommodates", "beds", "bedrooms", "availability_30", "availability_60"
    ]
    print(df[columns_to_describe].describe())
    
    grouped = df.groupby("neighbourhood_cleansed")[
        ["price", "number_of_reviews", "review_scores_rating"]
    ].agg(["mean", "median", "count"]).sort_values(("price", "mean"), ascending=False)
    print(grouped.head(10))
    
    room_stats = df.groupby("room_type")[
        ["price", "review_scores_rating", "availability_30"]
    ].agg(["mean", "median", "count"])
    print(room_stats)
    
    useful_numeric = df[columns_to_describe].dropna()
    sns.heatmap(useful_numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Between Key Airbnb Features")
    plt.show()
    
    pass
