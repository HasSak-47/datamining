import pandas as pd
from scipy.stats import ttest_ind, f_oneway, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from .utils import remove_outliers_iqr

def main():
    df = remove_outliers_iqr( pd.read_csv('listings_filtered.csv'), 'price')
    
    grouped_prices = [group["price"].values for _, group in df.groupby("room_type")]
    
    anova_stat, anova_p = f_oneway(*grouped_prices)
    print("🔬 ANOVA Test")
    print(f"F-statistic: {anova_stat:.4f}, p-value: {anova_p:.4e}")
    if anova_p < 0.05:
        print("➡️ Significant difference in price between room types (reject H0)\n")
    else:
        print("➡️ No significant difference (fail to reject H0)\n")
    
    room_types = df["room_type"].unique()
    if len(room_types) >= 2:
        prices_1 = df[df["room_type"] == room_types[0]]["price"]
        prices_2 = df[df["room_type"] == room_types[1]]["price"]
        t_stat, t_p = ttest_ind(prices_1, prices_2, equal_var=False)  # Welch's t-test
        print(f"🔬 T-Test Between '{room_types[0]}' and '{room_types[1]}'")
        print(f"T-statistic: {t_stat:.4f}, p-value: {t_p:.4e}")
        if t_p < 0.05:
            print("➡️ Significant difference in price between the two room types\n")
        else:
            print("➡️ No significant difference\n")
    
    kw_stat, kw_p = kruskal(*grouped_prices)
    print("🔬 Kruskal-Wallis Test (Non-parametric)")
    print(f"H-statistic: {kw_stat:.4f}, p-value: {kw_p:.4e}")
    if kw_p < 0.05:
        print("➡️ Significant difference in price between room types (reject H0)\n")
    else:
        print("➡️ No significant difference\n")
    
    sns.boxplot(data=df, x="room_type", y="price")
    plt.title("Price Distribution by Room Type (Outliers Removed)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
