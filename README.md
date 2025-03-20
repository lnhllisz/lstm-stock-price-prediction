# lstm-stock-price-prediction
**Data Preparation**

The dataset collected consists of six different Vietnamese stocks: ANV, DAT, HAH, KDC, NAF, and PVT. The selected features used for training the model include:
- Technical indicators: 200-day SMA, RSI, stochastic oscillators.
- Macroeconomic factors: GDP, interest rates.
- Target variable: Closing price of each stock.

Data preparation is of paramount importance in financial time-series forecasting since this is the phase when the dataset is properly formatted, clean, and optimized for deep learning models. The process begins with loading the dataset, where financial data is imported as a Pandas DataFrame. The 'Date' column is parsed as a datetime index which allows the model to recognize the sequential nature of the data. This is crucial for time-series forecasting.

The next step is when the dataset undergoes missing value handling to ensure data integrity. Financial datasets often contain missing values due to market holidays, reporting delays, or data collection issues. These gaps can distort predictions if not addressed properly. To handle missing values, there are some common techniques, namely forward-filling (using the last available data point), interpolation (estimating values based on surrounding data), or removing rows with excessive gaps. Identifying and addressing anomalies, such as extreme outliers caused by errors or market shocks, is also important to prevent the model from learning misleading patterns from data that do not conform to the usually expected behaviour . As a result, I align the macroeconomic data on a daily frequency as stock prices are believed to be available daily whilst GDP and interest rate data are reported less frequently (quarterly and monthly, respectively).

After cleaning the data, feature scaling is applied to standardize numerical values. Since financial data points, such as stock prices, interest rates, and trading volumes, can vary significantly in scale, the act of applying MinMaxScaler can ensure that all values fall within a consistent range, typically between 0 and 1. This data normalization step prevents large numbers from dominating the learning process and improves the stability and convergence speed of deep learning models. Such proper scaling is especially important for models like LSTMs and CNNs, which rely on patterns rather than absolute values. With the act of thoroughly preparing the dataset through three mentioned steps above, including structured loading, missing value handling, and feature scaling, a high-quality input yielded can accelerate the model’s ability to recognize trends, reduce errors, and improve financial forecasting accuracy at the same time, simultaneously.

Data preparation is then performed in two main steps: Data processing and normalization and Sequence generation. In the preprocessing step, the stock price data is re-ordered chronologically and filtered to the period from January 22, 2020, to January 14, 2025. Our inputs including technical and macroeconomic indicators are also processed. Technical indicators (SMA200, RSI, Stochastic Oscillator) are computed from the daily stock data. GDP data (with quarters formatted as “Q1 2020”, etc.) is converted by replacing the quarter identifier with a representative month–day (e.g., “Q1” → “-01-01”) and then merged with the daily stock data using forward-fill. Similarly, the monthly interest rate data is merged with the stock data by forward-filling the latest available value. Coming to the normalization and sequence generation step, selected features (Closing Price, RSI, SMA200, Stochastic Oscillator, GDP Index, and Interest Rate) are normalized to the range [0, 1] using a MinMaxScaler. To generate sequences, I use a sliding window (of 80 days), each sequence contains 80 consecutive days of features and the target is the closing price on the next day. The resulting data is reshaped into the three-dimensional format required by LSTM models, which is specified by (samples, time_steps, features).

**Model Configuration**

- Architecture:
  + Two LSTM layers:
    * First LSTM: return_sequences=True (passes output to the second LSTM layer).
    * Second LSTM: processes sequential data for prediction.
  + A dense layer with a single neuron predicts the next day’s closing price.

- Hyperparameters:
  + Hidden units: 50 per LSTM layer.
  + Number of layers: 2.
  + Batch size: 16.
  + Epochs: 50.
  + Output size: 1 (predicted closing price).
  + Optimizer: Adam.
  + Loss function: Mean Squared Error (MSE).

**Data Processing**

Before processing data, I split the dataset into training (70%), validation (15%), and testing (15%) sets to preserve temporal dependencies. For LSTM, the dataset is typically split into training, validation, and testing sets. Unlike statistical models like ARIMAX, LSTM requires a validation set because hyperparameter tuning is done through techniques like early stopping and monitoring loss curves, rather than relying solely on statistical criteria like AIC/BIC. The validation set helps assess how well the model generalizes during training, preventing overfitting and guiding adjustments to parameters like the learning rate, number of layers, and batch size (Ren, Zeng, Yang, & Urtasun, 2018; Ying, 2019). This makes validation essential in LSTM-based forecasting, whereas in ARIMAX, it is often optional. Therefore, our chosen ratio would be 70:15:15.

I trained an independent LSTM model for each stock. The training process involved feeding the model with time-series data, updating weights using the Adam optimizer, and minimizing the loss function (MSE). The training history was recorded, and the model performance was monitored using validation loss.

After training, the model’s performance is evaluated on the test set. Predictions are made and then inverse-transformed back to the original scale, the performance evaluation metrics such as MSE, RMSE, and Accuracy measure are then computed. Consequently, the predicted closing prices are compared against the actual values through plots, and error distributions are visualized.

