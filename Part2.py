import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler

assig_wine_data = pd.read_csv("C:/Users/user/PycharmProjects/EDA/ds_env/winequality-red.csv", sep=";")
#Data exploration
#PART 2
#Using .head, .tail to print the first and last 10 rows respectively
print("\n First 10 rows")
print(assig_wine_data.head(10))
print("\n Last 10 rows")
print(assig_wine_data.tail(10))
print("\n Dataset information:")
print(assig_wine_data.info())
print("\n Statistical information about the data set")
print(assig_wine_data.describe())

#Removing the white spaces from column headings and replacing them with the underscore _
assig_wine_data.columns = assig_wine_data.columns.str.replace(' ', '_').str.lower()

#using .isnull to check for missing values in each column and printing out the results
print("\n columns and their respective outputs for missing data: ")
print(assig_wine_data.isnull().sum())

#Cleaning
#-	Drop columns with object/string data types if not useful.
#-	Handle missing data (you may use .fillna(), .dropna(), or any imputation strategy).
#-	Remove duplicates if any.

#checking for duplicates in the data
print("\n Number of Duplicates in the dataset: ", assig_wine_data.duplicated().sum())
print("\n before duplicates: ", assig_wine_data.shape)

#Removing duplicates using inbuilt drop_duplicates
assig_wine_data.drop_duplicates(inplace=True)
print("\n after duplicates: ", assig_wine_data.shape)


# Handling any remaining missing values by filling with the median
#Using a for loop to iterate through the columns checking for any missing data
for col in assig_wine_data.select_dtypes(include=np.number).columns:
    if assig_wine_data[col].isnull().sum() > 0:             #if any column is found with missing data
        assig_wine_data[col] = assig_wine_data[col].fillna(assig_wine_data[col].median()) #use the .fillna function to fill it out with the median
    else:
        print(f"{col} has no missing values.") #Generate a print statement for each column to report that it had no missing values

# Identifying numerical columns for outlier detection
numerical_cols = assig_wine_data.select_dtypes(include=np.number).columns.tolist()
feats_for_outliers = [col for col in numerical_cols if col != 'quality'] #excluding 'quality' because it is a target column)
print(f"\nNumerical features for outlier detection: {feats_for_outliers}")  #Returning a list of columns which will be eligible for outlier checking

# Creating a copy of the dat for each method (iqr, z-score) I will use to check for outliers
assign_wine_iqr = assig_wine_data.copy()
assign_wine_zscore = assig_wine_data.copy()

#Function to find outliers using Outliers
def detect_outliers_iqr(assig_wine_data, feature):
  #Establish the 25th and 75th Quantiles
  Q1 = assign_wine_iqr[feature].quantile(0.25)
  Q3 = assign_wine_iqr[feature].quantile(0.75)
  #Calculating IQR
  IQR = Q3 - Q1
  #EStablishing Bounds
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  #Detect Outliers
  outliers = assign_wine_iqr[(assig_wine_data[feature] < lower_bound) | (assign_wine_iqr[feature] > upper_bound)]
  #Return Outliers
  return outliers

#Check for outliers in each feature
outlier_counts = {}  #creating a dictionary that will hold outlier dataframe
for col in assign_wine_iqr.columns[:-1]:    #iterating through the columns minus the target columns,
  outliers_iqr = detect_outliers_iqr(assign_wine_iqr, col) #calling the outlier function and storing output in the outlier_iqr variable
  outlier_counts[col] = outliers_iqr #The resulting DataFrame of outliers is stored in the outlier_counts dictionary with the column name as the key

# Collect all unique outlier row indices from the stored DataFrames
all_outlier_indices = []        #Creating an empty list to store indices of flagged outliers
for outlier_df in outlier_counts.values(): #Looping thru the DataFrames of outliers in the outlier_counts dictionary.
    all_outlier_indices.extend(outlier_df.index.tolist()) #extend adds these indices to the all_outlier_indices list.

#removing duplicates
#converting the list of indices to a set to remove any duplicate indices
# (a row could be an outlier in more than one feature) and then converts it back to a list.
unique_outlier_indices = list(set(all_outlier_indices))


# Create a new DataFrame excluding the outlier rows
#clean_assig_wine_iqr is created by dropping the rows with the unique_outlier_indices from the original assign_wine_iqr DataFrame
clean_assig_wine_iqr = assign_wine_iqr.drop(unique_outlier_indices)

print("\nOriginal DataFrame shape:", assign_wine_iqr.shape)
print("Cleaned DataFrame shape:", clean_assig_wine_iqr.shape)
print("Number of rows removed (outliers):", len(unique_outlier_indices))

# Z-score method on each feature in data set
zoutlier_counts = {}  #creating a dictionary to store outliers found in each column
for col in assign_wine_zscore.columns[:-1]:  #iterate through each column
  outliers = stats.zscore(assign_wine_zscore[col])  #each column calculates the Z-score for every data point using stats.zscore().
  zoutlier_counts[col] = len(outliers[(np.abs(outliers) > 3)])  #counts how many of these Z-scores have an abs value>3,
                                                                #indicating outliers based on the Z-score mthd.These counts are stored in the zoutlier_counts dictionary.
print("number of Z-score Outliers per feature: ")
print(pd.Series(zoutlier_counts).sort_values(ascending=False)) #Printing outliers by feature

# Identify outlier rows based on Z-scores
#checking for rows where the z score is > 3 storing them in the outlier_zscore_rows variable
outlier_zscore_rows = (np.abs(stats.zscore(assign_wine_zscore.drop(columns=['quality']))) > 3).any(axis=1)

# Create a new DataFrame without the outlier rows
#New data frame now excludes these outliers
clean_assig_wine_zscore = assign_wine_zscore[~outlier_zscore_rows].copy()

print("\nOriginal DataFrame shape:", assign_wine_zscore.shape)
print("Cleaned DataFrame shape (Z-score):", clean_assig_wine_zscore.shape)
print("Number of rows removed (Z-score outliers):", assign_wine_zscore.shape[0] - clean_assig_wine_zscore.shape[0])

#Create a new column sulfur_ratio = free_sulfur_dioxide / (total_sulfur_dioxide + 1e-6) and truncate it to 2 decimals.
#I am using the iqr generated data set without outliers to avoid the z-scores characteristic tendency to --
#assume that the data is a normal distribution.
clean_assig_wine_iqr['sulfur_ratio'] = (clean_assig_wine_iqr['free_sulfur_dioxide']/(clean_assig_wine_iqr['total_sulfur_dioxide']+0.000001)).round(2) #following the syntax, the square brackets hold the respective column names I am dealing with

#print(clean_assig_wine_iqr.head())

#Feature Engineering
#Applying MinMaxScaler and StandardScaler to numeric columns.
eligible_cols = clean_assig_wine_iqr.select_dtypes(include=np.number).columns.tolist()  # getting the numerical columns
features_to_scale = [col for col in numerical_cols if col != 'quality']  # eliminating Target column, quality

print("Red Wine Dataset ready for scaling. Features to consider:")
print(features_to_scale)

# Initialize an empty dictionary to store the skewness results
skewness_results = {}

# Loop through each identified numerical feature
for col in features_to_scale:
    # Calculate the skewness using scipy.stats.skew()
    # This function takes a 1-D array (a pandas Series in this case)
    skewness_value = skew(clean_assig_wine_iqr[col])

    # Store the result in our dictionary
    skewness_results[col] = skewness_value

# --- Step 3: Print the Skewness Results ---

print("\n--- Skewness of Numerical Features in Red Wine Dataset ---")
for col, val in skewness_results.items():
    # Print each feature's name and its calculated skewness, formatted to two decimal places
    print(f"- {col.replace('_', ' ').title()}: {val:.2f}")

# --- Optional: Identify and print highly skewed features (as a guideline) ---
skewness_threshold = 0.5  # A common threshold for considering a feature significantly skewed

highly_skewed_features = [col for col, val in skewness_results.items() if abs(val) > skewness_threshold]

print(f"\nFeatures identified as highly skewed (absolute skewness > {skewness_threshold}):")
if highly_skewed_features:
    for feature in highly_skewed_features:
        print(f"- {feature.replace('_', ' ').title()}")
else:
    print("None of the features are highly skewed based on the threshold.")


# --- Grouping Features Based on NEW Skewness Analysis ---
#putting skewed features in a list based on the output of the previous code
highly_skewed_features = ['residual_sugar', 'fixed_acidity', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'sulphates', 'alcohol'] # Use a plural name as it's a list

# All other features are relatively symmetrical
symmetrical_features = [col for col in features_to_scale if col not in highly_skewed_features]

print("\nApplying transformations and scalers to red wine dataset based on NEW skewness results:")

#creating a copy of the clean dataframe
clean_red_scaled = clean_assig_wine_iqr.copy()

# Initializing the scaler - Later to apply log transformation to improve the effectiveness of the scalar
scaler_for_log_transformed = StandardScaler()

for feature in highly_skewed_features:
    # Applying Log Transformation to the current feature
    # Use np.log1p (log(1+x)) to handle potential zero values gracefully
    clean_red_scaled[f'{feature}_log'] = np.log1p(clean_red_scaled[feature])
    print(f"  - Log-transformed: '{feature}' into '{feature}_log'")

    # Apply StandardScaler to the log-transformed feature
    clean_red_scaled[f'{feature}_log'] = \
        scaler_for_log_transformed.fit_transform(clean_red_scaled[[f'{feature}_log']])
    print(f"  - StandardScaler applied to log-transformed '{feature}'")

# Apply StandardScaler directly to Symmetrical Features
scaler_for_symmetrical = StandardScaler() # Initialize scaler ONCE
clean_red_scaled[symmetrical_features] = \
    scaler_for_symmetrical.fit_transform(clean_red_scaled[symmetrical_features])
print(f"  - StandardScaler applied directly to symmetrical features: {symmetrical_features}")


# --- Verification of Scaled Data (Descriptive Stats) ---
print("\n--- Describing scaled columns (sample examples) ---")

# Describe a sample highly skewed original and scaled column
print("\nOriginal 'residual_sugar' (Highly Skewed):")
print(clean_assig_wine_iqr['residual_sugar'].describe())
print("\nScaled 'residual_sugar_log' (Log then StandardScaled):")
print(clean_red_scaled['residual_sugar_log'].describe())

# Describe a sample symmetrical original and scaled column
print("\nOriginal 'ph' (Symmetrical):")
print(clean_assig_wine_iqr['ph'].describe())
print("\nScaled 'ph' (StandardScaled):")
print(clean_red_scaled['ph'].describe())

# Display the head of the DataFrame with new scaled columns
print("\nHead of the DataFrame with new scaled columns (sample):")
# Show original 'quality' plus a few original and their scaled versions
cols_to_show = ['quality'] + symmetrical_features[:3] + [f'{highly_skewed_features[0]}_log'] # Show one of the new log columns
print(clean_red_scaled[cols_to_show].head())


#Usinf the MinMaxScaler
red_minmax_scaled = clean_assig_wine_iqr.copy()
# Initialize the MinMaxScaler
minmax_scaler = MinMaxScaler()

# Applying the scaler to the selected features from the features in features_to_scale
# The fit_transform method of the minmax_scaler is applied to the columns specified in the features_to_scale list.
red_minmax_scaled[features_to_scale] = minmax_scaler.fit_transform(red_minmax_scaled[features_to_scale])

# Verifying the application of the MinMax scaler
print("\n--- Describing MinMax Scaled Features (Sample) ---")
for col in features_to_scale[:3]: # Print for the first 3 features
    print(f"\nStats for '{col}' (MinMax Scaled):")
    print(red_minmax_scaled[col].describe()) #the min value for these scaled features is close to 0 and the max value is close to 1

print("\nHead of the DataFrame with MinMax Scaled features (sample):")
# Show original 'quality' plus a few scaled features
cols_to_show_sample = ['quality'] + features_to_scale[:3]
print(red_minmax_scaled[cols_to_show_sample].head())    #showing the original quality column and a few of the scaled features.

#Add a new categorical column based on quality: quality_category = "low" for quality ≤ 4, medium for 5–6, high for ≥ 7
#Using the standard scaler scaled dataset for the following steps due to the MInMax scaler's sensitivity to outliers
clean_red_scaled['quality_category'] = pd.cut(clean_red_scaled['quality'], bins=[0, 4, 6, 10], labels=['low', 'median','high'])
print(clean_red_scaled.head(20))

#Group data by quality_category and compute the average of alcohol, pH, and sulphates.
print("Distribution of 'quality_category':")
print(clean_red_scaled['quality_category'].value_counts().sort_index())

#Grouping by 'quality_category' and calculating Averages - Used the groupby function
average_by_quality_category = clean_red_scaled.groupby('quality_category')[['alcohol', 'ph', 'sulphates']].mean()

#Use .pivot_table() to summarize alcohol against quality_category and sulphates.
print("\nAverage of Alcohol, pH, and Sulphates by Quality Category:")
print(average_by_quality_category)
