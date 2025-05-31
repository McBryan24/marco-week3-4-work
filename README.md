# Python for AI and Data Science: Week 3-4 Assignment - Wine Quality Analysis

## 1. Project Overview

This repository contains the solution for the Week 3-4 Assignment of the "Python for AI and Data Science" program. The primary goal of this assignment was to perform a comprehensive data cleaning and foundational analysis on the Red Wine Quality dataset.

The key tasks completed include:
* **Data Loading & Initial Inspection:** Loading the red wine dataset and understanding its structure and basic statistics.
* **Data Cleaning:**
    * Handling duplicate records.
    * Addressing missing values through imputation (using the median).
    * **Outlier Detection & Removal:** Applying both the Interquartile Range (IQR) method and Z-score method to identify and remove outliers from numerical features. A justified choice was made regarding which cleaned dataset to proceed with based on feature distributions.
* **Feature Engineering & Scaling:**
    * Applying `MinMaxScaler` and `StandardScaler` to numerical columns based on their distribution characteristics (skewness).
    * Performing log transformations on highly skewed features prior to scaling.
    * Using `pd.get_dummies()` for one-hot encoding of categorical columns (though in this specific red wine dataset, it primarily demonstrated the process for a single 'wine_type' category).
* **Exploratory Data Analysis (EDA) & Basic Statistical Analysis:**
    * Generating a correlation heatmap to visualize relationships between numerical features.
    * Performing a t-test to compare alcohol content between high and low-quality red wines.
    * Fitting a simple linear regression model (Alcohol → Quality) using NumPy and interpreting its coefficients.
    * Grouping data by custom quality categories and computing average chemical properties.
    * Summarizing data using `pivot_table()` for multi-dimensional insights.

## 2. How to Run the Code

To run the code from this assignment, follow these steps:

### 2.1. Prerequisites

Make sure you have Python installed (version 3.12 or higher recommended).
You will also need the following Python libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `scipy`

You can install them using pip:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy