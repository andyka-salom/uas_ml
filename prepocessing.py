# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Reading datasets
product_info = pd.read_csv('../Project_Uas ML Prak/product_info.csv')
reviews = pd.read_csv('../Project_Uas ML Prak/reviews_0-250.csv', dtype=str)

# Step 1: Display data
print("\nData yang telah dibaca (product_info):")
print(product_info.head())  # Menampilkan hanya 5 baris pertama
print("============================================================")

# Step 2: Detect missing values in product_info
print("\nDeteksi data product_info yang missing Value:")
missvalue_productinfo = product_info.isna().sum()  # Menggunakan .isna() untuk deteksi
print(missvalue_productinfo)
print("============================================================")

print("\nData yang telah dibaca (reviews):")
print(reviews.head())  # Menampilkan hanya 5 baris pertama
print("============================================================")

# Step 3: Detect missing values in reviews
print("\nDeteksi data reviews yang missing Value:")
missvalue_reviews = reviews.isna().sum()  # Menggunakan .isna() untuk deteksi
print(missvalue_reviews)
print("============================================================")


# Convert 'helpfulness' column to numeric
if 'helpfulness' in reviews.columns:
    reviews['helpfulness'] = pd.to_numeric(reviews['helpfulness'], errors='coerce')

# Handling Missing Values in product_info
product_info['rating'] = product_info['rating'].fillna(product_info['rating'].mean())
product_info['reviews'] = product_info['reviews'].fillna(0)
product_info['size'] = product_info['size'].fillna('Unknown')
product_info['variation_type'] = product_info['variation_type'].fillna('Not Specified')
product_info['variation_value'] = product_info['variation_value'].fillna('Not Specified')
product_info['highlights'] = product_info['highlights'].fillna('[]')
product_info['secondary_category'] = product_info['secondary_category'].fillna('Unknown')
product_info['tertiary_category'] = product_info['tertiary_category'].fillna('Unknown')
product_info['child_max_price'] = product_info['child_max_price'].fillna(product_info['child_max_price'].median())
product_info['child_min_price'] = product_info['child_min_price'].fillna(product_info['child_min_price'].median())

# Dropping columns with too many missing values
product_info = product_info.drop(columns=['variation_desc', 'ingredients', 'value_price_usd', 'sale_price_usd'])

# Handling Missing Values in reviews
reviews['is_recommended'] = reviews['is_recommended'].fillna(0)
reviews['review_text'] = reviews['review_text'].fillna('No review provided')
reviews['review_title'] = reviews['review_title'].fillna('Untitled')
reviews['skin_tone'] = reviews['skin_tone'].fillna('Unknown')
reviews['eye_color'] = reviews['eye_color'].fillna('Unknown')
reviews['skin_type'] = reviews['skin_type'].fillna('Unknown')
reviews['hair_color'] = reviews['hair_color'].fillna('Unknown')
reviews['helpfulness'] = reviews['helpfulness'].fillna(reviews['helpfulness'].median())

# Step 4: Detecting Outliers
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

print("\nDetecting outliers in product_info:")
for col in ['rating', 'reviews', 'child_max_price', 'child_min_price']:
    if col in product_info.columns and pd.api.types.is_numeric_dtype(product_info[col]):
        outliers = detect_outliers_iqr(product_info, col)
        print(f"{col}: Found {len(outliers)} outliers")

print("\nDetecting outliers in reviews:")
for col in ['rating', 'helpfulness']:
    if col in reviews.columns and pd.api.types.is_numeric_dtype(reviews[col]):
        outliers = detect_outliers_iqr(reviews, col)
        print(f"{col}: Found {len(outliers)} outliers")

# Step 5: Normalization and Encoding
encoder = LabelEncoder()
for col in ['size', 'variation_type', 'variation_value', 'secondary_category', 'tertiary_category']:
    product_info[col] = encoder.fit_transform(product_info[col])

for col in ['skin_tone', 'eye_color', 'skin_type', 'hair_color']:
    reviews[col] = encoder.fit_transform(reviews[col])

# Step 6: Merging Datasets for Recommendation System
merged_data = pd.merge(reviews, product_info, on='product_id', how='inner')

# # Step 7: Saving Preprocessed Data
merged_data.to_csv('../Project_Uas ML Prak/preprocessed_data_with_outliers.csv', index=False)
print("Preprocessed data saved to ../Project_Uas ML Prak/preprocessed_data_with_outliers.csv")
