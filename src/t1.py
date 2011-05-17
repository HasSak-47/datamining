import pandas as pd

def main():
    df = pd.read_csv("listings.csv")
    
    useful_columns = [
        "id", "name", "description", "neighbourhood_cleansed", 
        "latitude", "longitude",
        
        "host_id", "host_name", "host_since", "host_is_superhost",
        
        "property_type", "room_type", "accommodates", 
        "bathrooms_text", "bedrooms", "beds", "amenities",
        
        "price", "minimum_nights", "maximum_nights", 
        "has_availability", "availability_30", "availability_60", "availability_90",
        
        "number_of_reviews", "first_review", "last_review", 
        "review_scores_rating", "review_scores_cleanliness", # pyright:ignore
        
        "instant_bookable", "calculated_host_listings_count"
    ]
    
    df_filtered = df[useful_columns]

    df_filtered.loc[:,'price'] = df_filtered['price'].replace('[$,]', '', regex=True).astype(float)
    df_filtered.loc[:, "host_since"] = pd.to_datetime(df_filtered["host_since"])
    df_filtered.loc[:, "last_review"] = pd.to_datetime(df_filtered["last_review"])
    
    df_filtered.to_csv("listings_filtered.csv", index=False)
