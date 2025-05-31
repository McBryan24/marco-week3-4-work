from sklearn.preprocessing import PowerTransformer
from Part2 import *

#Calculate skewness and kurtosis for scaled numeric columns.
# Skewness Results
print("\nSkewness of Numerical Features in Red Wine Dataset")
for col, val in skewness_results.items():
    # Print each feature's name and its calculated skewness, formatted to two decimal places
    print(f"- {col.replace('_', ' ').title()}: {val:.2f}")

#Kurtosis
red_kurtosis = clean_assig_wine_iqr.kurtosis()
print("\nKurtosis of Numerical Features in Red Wine Dataset")
for col, val in red_kurtosis.items():
    # Print each feature's name and its calculated skewness, formatted to two decimal places
    print(f"- {col.replace('_', ' ').title()}: {val:.2f}")


#Use a histogram and density plots to visualize skewed features before and after transformation.
skewed_features = ['residual_sugar', 'fixed_acidity', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'sulphates',
                   'alcohol'] #Skewed Features as obtained in Part2

# Apply Yeo-Johnson -- > negative/zero values
pt = PowerTransformer(method='yeo-johnson') #Yeo-Johnson - as provided in the sklearn.preprocessing
transformed_data = clean_red_scaled.copy()  #Create a copy first name it Transformed_data
transformed_data[skewed_features] = pt.fit_transform(transformed_data[skewed_features]) #run the transformer on the skewed features

# Comparison of before and after transformation
if len(skewed_features) == 1: #code block runs if only one skewed feature is detected
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    col = skewed_features[0]
    sns.histplot(clean_assig_wine_iqr[col], kde=True, ax=axes[0], color='blue', alpha=0.7).set_title(
        f'Before Transformation - {col}')
    sns.histplot(transformed_data[col], kde=True, ax=axes[1], color='green', alpha=0.7).set_title(
        f'After Transformation - {col}')
else:
    # If there are multiple skewed features,
    fig, axes = plt.subplots(len(skewed_features), 2, figsize=(14, 4 * len(skewed_features)))
    for i, col in enumerate(skewed_features):
        sns.histplot(clean_assig_wine_iqr[col], kde=True, ax=axes[i, 0], color='blue', alpha=0.7).set_title(
            f'Before Transformation - {col}')
        sns.histplot(transformed_data[col], kde=True, ax=axes[i, 1], color='green', alpha=0.7).set_title(
            f'After Transformation - {col}')
        # Removed the duplicate line for 'Before Transformation'

plt.tight_layout()
plt.show()


from scipy.stats import ttest_ind # For independent samples t-test
#print(transformed_data.head(30))
low_quality_wine = transformed_data[transformed_data['quality'] <= 6] #Categorizing alcohol quality below 6 as low
high_quality_wine = transformed_data[transformed_data['quality'] >= 7] #Categorizing alcohol quality above 7 as high

#Extracting Alcohol Content for Each Group
alcohol_low_quality = low_quality_wine['alcohol']
alcohol_high_quality = high_quality_wine['alcohol']
# Checking means and standard deviations for each group
print(f"Mean alcohol in Low Quality wines: {alcohol_low_quality.mean():.2f}")
print(f"Std dev alcohol in Low Quality wines: {alcohol_low_quality.std():.2f}")
print(f"Mean alcohol in High Quality wines: {alcohol_high_quality.mean():.2f}")
print(f"Std dev alcohol in High Quality wines: {alcohol_high_quality.std():.2f}\n")

# Performing the Independent Samples t-test
t_statistic, p_value = ttest_ind(alcohol_high_quality, alcohol_low_quality, equal_var=False) # ttest_ind performs an independent two-sample t-test.

print(f"T-test Results (High Quality vs. Low Quality Alcohol Content):")
print(f"T-statistic: {t_statistic:.3f}")
print(f"P-value: {p_value:.3f}")

# Interpreting Results
alpha = 0.05 # Standard significance level

if p_value < alpha:
    print(f"\nSince the p-value ({p_value:.3f}) is less than the significance level ({alpha}),")
    print("we reject the null hypothesis.")
    print("Conclusion: There is a statistically significant difference in the mean alcohol content between high-quality red wines and low-quality red wines.")

else:
    print(f"\nSince the p-value ({p_value:.3f}) is greater than the significance level ({alpha}),")
    print("we fail to reject the null hypothesis.")
    print("Conclusion: There is no statistically significant difference in the mean alcohol content between high-quality red wines and low-quality red wines based on this test.")


#Extracting X (alcohol) and y (quality) values
X = transformed_data['alcohol'].values
y = transformed_data['quality'].values

print(f"X (alcohol) shape: {X.shape}")
print(f"y (quality) shape: {y.shape}")

#Implement Simple Linear Regression using NumPy
# Simple Linear Regression Equation: y = b0 + b1 * X
# b1 (slope) = sum((Xi - mean(X)) * (Yi - mean(Y))) / sum((Xi - mean(X))^2)
# b0 (intercept) = mean(Y) - b1 * mean(X)

# Calculating the means using the mean() function
mean_X = np.mean(X)
mean_y = np.mean(y)

# Calculating b1 (slope)
numerator = np.sum((X - mean_X) * (y - mean_y)) #following the equation written above
denominator = np.sum((X - mean_X)**2) #following the equation written above
b1 = numerator / denominator

# Calculate b0 (intercept)
b0 = mean_y - b1 * mean_X

print(f"\nCalculated Coefficients:")
print(f"Slope (b1): {b1:.4f}")
print(f"Intercept (b0): {b0:.4f}")

# Interpreting the Coefficients
print("\nInterpretation of Coefficients ")
print(f"The linear regression model is: Quality = {b0:.4f} + {b1:.4f} * Alcohol")

print(f"\nIntercept (b0): {b0:.4f}")
print("The intercept represents the predicted quality of a red wine when its alcohol content is 0%.")
print("An alcohol content of 0% may not meaningful, so the intercept here is understood as a mathematical constant in the regression equation meant to adjust the line's position on the y-axis.")

print(f"\n**Slope (b1): {b1:.4f}**")
print("For every increase in alcohol content, the predicted wine quality score is estimated to increase")
print("This suggests a positive linear relationship: as alcohol content increases, wine quality tends to increase.")

