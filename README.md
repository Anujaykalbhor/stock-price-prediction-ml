#  Stock Price Prediction: LSTM vs Ensemble Methods

## Project Overview

This repository presents a comprehensive comparative analysis of three machine learning approaches for predicting stock closing prices using historical market data. The project implements and evaluates Long Short-Term Memory (LSTM) neural networks against ensemble methods including Random Forest and XGBoost to determine the most effective model for financial time series forecasting.

##  Objective

The primary goal is to develop and compare predictive models that can accurately forecast stock closing prices based on historical trading data. Through rigorous experimentation and evaluation, this project identifies the optimal approach for stock price prediction while providing insights into the strengths and limitations of different machine learning paradigms in financial forecasting.

##  Dataset & Features

The analysis utilizes a comprehensive stock market dataset containing multiple securities with the following features:
- **Temporal Data**: Date-indexed time series spanning multiple years
- **OHLCV Data**: Open, High, Low, Close prices and trading Volume
- **Multi-Stock Coverage**: Analysis across different stock symbols
- **Feature Engineering**: Advanced technical indicators and statistical measures

##  Methodology

### Exploratory Data Analysis
- **Yearly Trend Analysis**: Comprehensive examination of year-end closing prices for each stock
- **Temporal Pattern Recognition**: Identification of seasonal trends and cyclical behaviors
- **Statistical Profiling**: Distribution analysis and correlation studies
- **Visualization**: Interactive plots showing long-term price movements and volatility patterns

### Feature Engineering
The project implements sophisticated feature engineering techniques:
- **Lagged Variables**: Close_lag_1, Close_lag_3, Close_lag_7 for capturing temporal dependencies
- **Technical Indicators**: Daily returns, 5-day rolling means and standard deviations
- **Volatility Measures**: 10-day rolling standard deviation for risk assessment
- **Volume Analysis**: Percentage changes in trading volume as momentum indicators
- **Statistical Features**: Moving averages and volatility-based technical indicators

### Model Architectures

#### 1. LSTM Neural Network
- **Deep Learning Approach**: Three-layer LSTM architecture with 50 units each
- **Sequence Learning**: 100-timestep input sequences with 5 features
- **Temporal Memory**: Designed to capture long-term dependencies in stock price movements
- **Optimization**: Adam optimizer with Mean Squared Error loss function

#### 2. Random Forest
- **Ensemble Method**: Hyperparameter-tuned using RandomizedSearchCV
- **Feature Importance**: Automatic feature selection and importance ranking
- **Overfitting Prevention**: Built-in regularization through ensemble averaging
- **Tabular Data Processing**: Optimized for engineered feature sets

#### 3. XGBoost
- **Gradient Boosting**: Advanced boosting algorithm with extensive hyperparameter tuning
- **Performance Optimization**: Subsample and column sampling for better generalization
- **Regularization**: L1 and L2 regularization to prevent overfitting
- **Scalability**: Efficient handling of large datasets with parallel processing

##  Results & Performance

### Quantitative Metrics
- **LSTM Performance**: Achieved exceptional results with R² = 0.98 on test data and RMSE ≈ 0.005
- **Random Forest**: R² = 0.22 with RMSE ≈ 21.0, indicating moderate predictive capability
- **XGBoost**: R² = 0.20 with RMSE ≈ 22.0, similar performance to Random Forest
- **Evaluation Metrics**: Comprehensive assessment using RMSE, MAE, and R² scores

### Key Insights
The LSTM model demonstrated superior performance in capturing the sequential nature of stock price movements, significantly outperforming traditional ensemble methods. The deep learning approach proved particularly effective at learning complex temporal patterns and non-linear relationships inherent in financial time series data.

##  Technical Implementation

- **Programming Language**: Python 3.x with comprehensive data science stack
- **Deep Learning**: Keras/TensorFlow for LSTM implementation
- **Machine Learning**: scikit-learn for ensemble methods and evaluation
- **Data Processing**: pandas, NumPy for data manipulation and feature engineering
- **Visualization**: matplotlib for comprehensive exploratory data analysis
- **Optimization**: RandomizedSearchCV for hyperparameter tuning

## Future Enhancements

- **Advanced Architectures**: Implementation of Transformer-based models for sequence-to-sequence learning
- **External Data Integration**: Incorporation of news sentiment analysis and macroeconomic indicators
- **Model Interpretability**: SHAP values and feature importance analysis for better understanding
- **Cross-Validation**: Time series cross-validation for more robust model evaluation
- **Real-time Prediction**: Development of streaming prediction pipeline for live market data

##  Academic Contribution

This project serves as a comprehensive case study in financial machine learning, demonstrating the comparative effectiveness of different modeling approaches for time series forecasting in volatile financial markets. The methodology and results provide valuable insights for both academic research and practical applications in quantitative finance.
