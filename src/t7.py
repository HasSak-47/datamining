import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from .utils import remove_outliers_iqr

def main():
    df = pd.read_csv('listings_filtered.csv')

    features = ["review_scores_rating", "number_of_reviews"]
    df_kmeans = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_kmeans)
    
    kmeans = KMeans(n_clusters=2, random_state=0)
    df_kmeans["cluster"] = kmeans.fit_predict(X_scaled)
    
    sns.scatterplot(x="review_scores_rating", y="number_of_reviews", hue="cluster", data=df_kmeans)
    plt.title("K-Means Clustering")
    plt.show()

