# Diamond Price Prediction Using Regression Models

This project explores the prediction of diamond prices using various regression techniques. The analysis includes Simple Linear Regression (SLR), Multiple Linear Regression (MLR), and Quantile Regression, leveraging both numerical and categorical features. We use the **diamonds.csv** dataset from Kaggle, which contains detailed information about diamonds, such as their weight (carat), cut, color, clarity, and dimensions (x, y, z), along with their price.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Data Overview](#data-overview)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

The goal of this project is to predict the price of diamonds using various regression techniques. We implement Simple Linear Regression, Multiple Linear Regression (with and without categorical variables), and Quantile Regression. The dataset contains both numerical and categorical variables, and we explore the effect of including these variables on the model's accuracy.

## Features

- **Descriptive Analysis**: Investigate the structure of the dataset, check for missing values, and understand the distribution of variables.
- **Correlation Heatmap**: Visualize correlations between the diamond price and other numerical features.
- **Simple Linear Regression**: Predict diamond prices using a single numerical feature (carat).
- **Multiple Linear Regression**: Predict prices using multiple numerical features and explore the impact of adding categorical variables like cut, color, and clarity.
- **Quantile Regression**: Predict diamond prices based on the median (50th percentile), a more robust estimate than the mean.
- **Performance Metrics**: Evaluate the models using R-squared, Mean Absolute Error (MAE), and the ratio of MAE to the average price.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rahulpudurkar/Diamond-Price-Prediction-Using-Regression-Models.git
   cd diamond-price-prediction
   ```

2. Install the required Python libraries.
  
3. Ensure the `diamonds.csv` dataset is in the working directory. You can download it from Kaggle or use the one provided in the repository.

## Data Overview

The dataset contains the following features:
- **carat**: Weight of the diamond.
- **cut**: Quality of the diamond cut (Fair, Good, Very Good, Premium, Ideal).
- **color**: Diamond color, from D (best) to J (worst).
- **clarity**: Measure of diamond clarity (I1 to IF).
- **price**: Price of the diamond in USD.
- **x, y, z**: Length, width, and depth of the diamond in mm.
- **depth**: Total depth percentage.
- **table**: Width of the top of the diamond relative to its widest point.

## Usage

### 1. Exploratory Data Analysis
- Load and inspect the data:
  ```python
  diamonds_data = pd.read_csv('diamonds.csv')
  print(diamonds_data.head())
  ```

### 2. Simple Linear Regression
- Split the data based on the carat and price variables:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(diamonds_data[['carat']], diamonds_data['price'], test_size=0.25, random_state=50)
  ```

- Fit and evaluate a linear regression model:
  ```python
  model = LinearRegression()
  model.fit(X_train, y_train)
  ```

### 3. Multiple Linear Regression
- Fit a model using multiple features (carat, depth, table, x, y, z):
  ```python
  model_multi = LinearRegression()
  model_multi.fit(X_train_multi, y_train_multi)
  ```

### 4. Quantile Regression
- Perform quantile regression for predicting the median price:
  ```python
  import statsmodels.formula.api as smf
  formula = "price ~ carat + depth + table + x + y + z + cut + color + clarity"
  mod = smf.quantreg(formula, data=final_df)
  res = mod.fit(q=0.5)
  ```

## Models

1. **Simple Linear Regression**: Predicts diamond prices using only one feature (carat).
2. **Multiple Linear Regression (without categorical variables)**: Uses multiple numerical features to predict diamond prices.
3. **Multiple Linear Regression (with categorical variables)**: Adds categorical variables such as cut, color, and clarity using one-hot encoding.
4. **Quantile Regression**: Predicts the median price to handle outliers more robustly.

## Results

### Performance Metrics
- **R-Squared**: Evaluates the goodness of fit for the models. Higher values indicate better fits.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **Fraction of MAE to Average Price**: Provides a normalized view of the MAE relative to the average price.

### Key Findings
- The **Multiple Linear Regression model** with categorical variables performs better than the model without them.
- **Quantile Regression** results in lower Mean Absolute Error, showcasing its robustness against outliers.

## Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.
--- 

