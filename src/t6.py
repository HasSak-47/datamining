import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def main():
    df = pd.read_csv('listings_filtered.csv')
    X = df[["price", "number_of_reviews"]]
    le = LabelEncoder()
    y = le.fit_transform(df["room_type"])
    class_names = le.classes_
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print("KNN Accuracy:", accuracy_score(y_test, predictions))
    
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', yticklabels=class_names, xticklabels=class_names)
    plt.title("Confusion Matrix for KNN Classifier")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
