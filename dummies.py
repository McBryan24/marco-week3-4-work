from Part2 import *
#Using getdummies()
# Load the datasets
red = pd.read_csv("C:/Users/user/PycharmProjects/EDA/ds_env/winequality-red.csv", sep=";")
white = pd.read_csv("C:/Users/user/PycharmProjects/EDA/ds_env/winequality-red.csv", sep=";")

# Adding a 'wine_type' column to distinguish the datasets

red['wine_type'] = 'red'
white['wine_type'] = 'white'

# Concatenate the datasets
wine = pd.concat([red, white], ignore_index=True)

# Identify the categorical column(s) to encode
categorical_cols = ['wine_type'] # In this combined dataframe we are going to use 'wine_type'

# Applying one-hot encoding
# drop_first=False is used here to keep both 'wine_type_red' and 'wine_type_white' columns.
# If drop_first=True, it would create only one (e.g., 'wine_type_white'), where 0 would imply 'red'.
combined_wine_encoded = pd.get_dummies(wine, columns=categorical_cols, drop_first=False)

print("\n--- Combined Wine DataFrame AFTER get_dummies() ---")
print("Shape:", combined_wine_encoded.shape) # Notice the shape increased by 1 column
print("Head (showing new one-hot encoded columns):")
# Now we'll see 'wine_type_red' and 'wine_type_white' columns
print(combined_wine_encoded[['wine_type_red', 'wine_type_white', 'alcohol']].head())
print("Tail (showing new one-hot encoded columns):")
print(combined_wine_encoded[['wine_type_red', 'wine_type_white', 'alcohol']].tail())