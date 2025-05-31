import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler
from Part2 import *

#Heatmap for correlation (drop non-numeric columns first).
#Boxplot of alcohol by quality_category.
#Histogram of pH with bins=20.

plt.figure(figsize=(10, 6))
sns.heatmap(clean_red_scaled.select_dtypes(include=np.number).drop(columns=['residual_sugar_log', 'fixed_acidity_log', 'alcohol_log', 'free_sulfur_dioxide_log', 'total_sulfur_dioxide_log', 'sulphates_log']).corr(),
            annot=True, cmap='coolwarm', fmt = '.2f')
plt.title('Correlation Heatmap')
plt.show()

#Boxplot of alcohol by quality_category.
sns.boxplot(x='alcohol', y='quality_category', data=clean_red_scaled)
plt.title('Distribution of Alcohol by Quality Category')
plt.xlabel('Alcohol')
plt.ylabel('Quality category')
plt.show()

#Histogram of pH with bins=20
plt.figure(figsize=(10, 6))
plt.hist(clean_red_scaled['ph'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of pH')
plt.xlabel('pH')
plt.ylabel('Frequency')
plt.show()

#PairGrid plot for: alcohol, volatile_acidity, sulphates,
#total_sulfur_dioxide (Hue quality_category)
data = sns.PairGrid(
    clean_red_scaled,
    vars = ['alcohol', 'volatile_acidity', 'sulphates', 'total_sulfur_dioxide'],
    hue = 'quality_category',
    palette = 'viridis'

)

data.map_upper(sns.regplot, scatter_kws={'s': 10}, line_kws={'color': 'red'}) # Using regplot for scatter + regression line
data.map_lower(sns.kdeplot)   #Generating the kdeplots
data.map_diag(sns.histplot, kde=True)       #Generating the histplots
plt.show()