
 # FinBERT-LSTM Stock Market Prediction
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</div>

## 📊 Project Overview

This project combines Natural Language Processing (NLP) and Deep Learning to predict stock market movements based on financial news sentiment. By leveraging FinBERT for financial news sentiment analysis and LSTM neural networks for time series forecasting, we achieve improved accuracy in stock price predictions.

The system collects financial news from the New York Times, analyzes its sentiment using FinBERT (a financial domain-specific BERT model), and combines this with historical stock price data to predict future market trends.

## 🎯 Features

- **Financial News Collection**: Automated retrieval of relevant news from the New York Times API
- **Stock Data Acquisition**: Historical stock price data collection from Yahoo Finance
- **Sentiment Analysis**: Advanced financial sentiment scoring using FinBERT
- **Multiple Predictive Models**:
  - Standard MLP (Multi-Layer Perceptron)
  - LSTM (Long Short-Term Memory) neural network
  - Enhanced FinBERT-LSTM model integrating sentiment features
- **Performance Comparison**: Comprehensive evaluation metrics to compare model performance

## 📈 Results

Our comparative analysis demonstrates that the FinBERT-LSTM model outperforms traditional models:

| Model | MAE | MAPE | Accuracy |
|-------|-----|------|----------|
| MLP | 218.33 | 0.0177 | 98.23% |
| LSTM | 180.58 | 0.0146 | 98.54% |
| FinBERT-LSTM | 153.72 | 0.0121 | 98.79% |

This shows the value of incorporating financial news sentiment into stock market prediction models.

## 🚀 Project Structure

```
├── 1_news_collection.py       # Collects financial news from NY Times API
├── 2_stock_data_collection.py # Downloads stock price data from Yahoo Finance
├── 3_news_data_cleaning.py    # Cleans and aligns news data with stock prices
├── 4_news_sentiment_analysis.py # Performs sentiment analysis using FinBERT
├── 5_MLP_model.py             # Implements a Multi-Layer Perceptron model
├── 6_LSTM_model.py            # Implements a standard LSTM model
├── 7_lstm_model_bert.py       # Implements an enhanced LSTM model with FinBERT features
├── analysis.py                # Data analysis and visualization utilities
├── bertmodel.keras            # Pre-trained FinBERT-LSTM model
├── lstm_model.h5              # Pre-trained LSTM model
├── Lstm + Finbert.ipynb       # Jupyter notebook with complete workflow
├── news_data.csv              # Processed news data
├── news_data1.csv             # Additional processed news data
├── news.csv                   # Raw news data
├── news1.csv                  # Additional raw news data
├── sentiment.csv              # News with sentiment scores
├── sentiment1.csv             # Additional news sentiment data
├── stock_price.csv            # Historical stock price data
└── stock_price1.csv           # Additional stock price data
```

## 🛠️ Technology Stack

- **Python**: Core programming language
- **TensorFlow & Keras**: Deep learning framework for neural network models
- **Transformers**: Hugging Face library for FinBERT implementation
- **Pandas & NumPy**: Data manipulation and processing
- **Matplotlib**: Data visualization
- **yfinance**: Yahoo Finance API wrapper for stock data
- **PyNYTimes**: New York Times API client for news collection

## 📋 Requirements

```
tensorflow>=2.7.0
pandas
numpy
scikit-learn
matplotlib
transformers
pynytimes
yfinance
nltk
```

## 🔧 Installation & Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/FinBERT-LSTM.git
   cd FinBERT-LSTM
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API access:
   - Obtain a New York Times API key from [developer.nytimes.com](https://developer.nytimes.com/)
   - Replace the API key in `1_news_collection.py`

## 📊 Usage

### Data Collection & Processing

1. Collect financial news:
   ```
   python 1_news_collection.py
   ```

2. Download stock price data:
   ```
   python 2_stock_data_collection.py
   ```

3. Clean and align news data:
   ```
   python 3_news_data_cleaning.py
   ```

4. Perform sentiment analysis:
   ```
   python 4_news_sentiment_analysis.py
   ```

### Model Training & Evaluation

5. Train and evaluate models:
   ```
   python 5_MLP_model.py
   python 6_LSTM_model.py
   python 7_lstm_model_bert.py
   ```

### Running the Complete Workflow

- Open and run the Jupyter notebook:
  ```
  jupyter notebook "Lstm + Finbert.ipynb"
  ```

## 📈 Future Market Prediction

The project includes functionality to predict future market movements based on current news and historical data:

```python
# Example of using the prediction functionality
from analysis import analyze_market_future
lstm_preds, bert_preds = analyze_market_future(days_ahead=5)
```

## 📝 Citation

If you use this project in your research or work, please consider citing:

```
@software{FinBERT_LSTM,
  author = {Harsh Maheshwari , Harsh Bhanushali},
  title = {FinBERT-LSTM: Stock Market Prediction Using Financial News Sentiment},
  year = {2025},
  url = {https://github.com/yourusername/FinBERT-LSTM}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">
  <i>Note: This project is for educational purposes only and should not be used for financial advice or trading decisions.</i>
</div>
