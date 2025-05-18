import pandas as pd

def main():
    df = pd.read_csv('listings.csv')
    
    useful_columns = [
        'id', 'name', 'neighbourhood_cleansed', 
        'latitude', 'longitude',
        
        'host_id', 'host_since', 'host_is_superhost',
        
        'property_type', 'room_type', 'accommodates', 
        'bathrooms_text', 'bedrooms', 'beds', 'amenities',
        
        'price',
        
        'number_of_reviews', 'first_review', 'last_review', 
        'review_scores_rating', 'review_scores_cleanliness',
    ]
    
    df_filtered = df[useful_columns]

    df_filtered.loc[:,'price'] = df_filtered['price'].replace('[$,]', '', regex=True).astype(float) # pyright:ignore
    df_filtered.loc[:, 'host_since'] = pd.to_datetime(df_filtered['host_since'])
    df_filtered.loc[:, 'last_review'] = pd.to_datetime(df_filtered['last_review'])
    df_filtered = df_filtered.dropna()
    
    df_filtered.to_csv('listings_filtered.csv', index=False)
