import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def calculate_metrics(k, X):
    print(f'k: {k}')
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=4)
    labels = kmeans.fit_predict(X)
    return {
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': silhouette_score(X, labels)
    }

def optimal_clusters(X, max_k=10):
    K_range = range(2, max_k+1)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(partial(calculate_metrics, X=X), K_range))
    
    distortions = [r['inertia'] for r in results]
    silhouette_scores = [r['silhouette'] for r in results]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method')
    
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    
    plt.tight_layout()
    plt.show()
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters suggested: {optimal_k}")
    return optimal_k

def main():
    df = pd.read_csv('listings_filtered.csv')
    
    features = ["review_scores_rating", "number_of_reviews"]
    df_kmeans = df[features].dropna()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_kmeans)
    
    optimal_k = optimal_clusters(X_scaled)
    
    pipeline = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    )
    df_kmeans["cluster"] = pipeline.fit_predict(df_kmeans[features])
    
    # Visualization (same as before)
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        x="review_scores_rating", 
        y="number_of_reviews", 
        hue="cluster", 
        data=df_kmeans,
        palette="viridis",
        alpha=0.7
    )
    
    centers = pipeline.named_steps['kmeans'].cluster_centers_
    centers = scaler.inverse_transform(centers)
    plt.scatter(
        centers[:, 0], 
        centers[:, 1], 
        c='red', 
        s=200, 
        alpha=0.8, 
        marker='X',
        label='Cluster Centers'
    )
    
    plt.title(f"K-Means Clustering (k={optimal_k})")
    plt.xlabel("Review Scores Rating")
    plt.ylabel("Number of Reviews")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nCluster Statistics:")
    print(df_kmeans.groupby('cluster')[features].mean())
    
