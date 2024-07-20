# Stock Price Prediction Using Deep Learning

## Overview

This project aims to predict stock prices using advanced machine learning techniques. Although primarily designed for the Indian stock market, the code can also be applied to international markets. The project involves collecting historical stock data, preprocessing it, training various machine learning models, and evaluating their performance to make accurate predictions.

## Importance

Stock price prediction is a crucial aspect of financial markets, allowing traders, investors, and financial analysts to make informed decisions. Accurate predictions can lead to profitable trades and investments, minimizing risks and maximizing returns.

## Features

1. **Data Collection**: Automatically fetch historical stock data using Yahoo Finance.
2. **Feature Engineering**: Preprocess data, extract relevant features, and engineer new features that might be predictive of stock prices, such as technical indicators and sentiment analysis.
3. **Model Selection**: Implement and compare various machine learning models, including linear regression, decision trees, random forests, support vector machines, and advanced techniques like LSTMs.
4. **Training and Evaluation**: Train models on historical data, optimize hyperparameters, prevent overfitting, and evaluate performance using metrics like MSE, RMSE, and MAE.
5. **Regular Updates**: The model can be regularly updated with new data to adapt to changing market conditions.

## Technologies Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib** and **Seaborn**: Data visualization
- **yfinance**: Fetching stock data from Yahoo Finance
- **scikit-learn**: Machine learning library
- **TensorFlow/Keras**: Deep learning framework

## How to Use This Code

### Prerequisites

- Python 3.x
- Git

### Installation

1. **Fork the Repository**

   Fork the repository to your own GitHub account by clicking the "Fork" button on the top right of the repository page.

2. **Clone the Repository**

Clone the forked repository to your local machine:

```sh
git clone https://github.com/ShivamGupta92/StockPrice_Prediction-using-deep-learning.git
```
3. Changeing working directory
```sh
cd StockPrice_Prediction-using-deep-learning
 ```
   
4. Install Requirements

Install the necessary packages by running:
```sh
pip install -r requirements.txt
 ```

### Running the Code
- Prepare Data
- Train the Model

Train the model using the provided script:
```sh
python train_model.py
 ```

- Make Predictions
Use the trained model to make predictions:

```sh
python predict.py
 ```

### Future Work
- Incorporate sentiment analysis of news articles to enhance predictive accuracy.
- Explore additional technical indicators and economic indicators for feature engineering.
- Implement an automated pipeline for regular model updates and re-training.

## Author
Shivam Gupta

## LICENSE
This project is licensed under the [MIT License](LICENSE).

