
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np 

sp500_ticker = yf.Ticker("^GSPC")

sp500 = sp500_ticker.history(period="max")

print("--- S&P 500 Data (First 5 Rows) ---")
print(sp500.head())
print("\n--- S&P 500 Data (Last 5 Rows) ---")
print(sp500.tail())

sp500.plot.line(y="Close", use_index=True, title="S&P 500 Closing Price History")


del sp500["Dividends"]
del sp500["Stock Splits"]

print("\n--- Data after removing unnecessary columns ---")
print(sp500.head())

sp500["Tomorrow"] = sp500["Close"].shift(-1)

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()

print("\n--- Data with 'Tomorrow' and 'Target' columns (from 1990) ---")
print(sp500.head())

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]

test = sp500.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]

model.fit(train[predictors], train["Target"])

preds = model.predict(test[predictors])

preds = pd.Series(preds, index=test.index)

initial_precision = precision_score(test["Target"], preds)

print(f"\nInitial Model Precision: {initial_precision:.4f}")

combined = pd.concat([test["Target"], preds], axis=1)
combined.columns = ['Actual', 'Predicted']
print("\n--- Actual vs. Predicted (Initial Model) ---")
print(combined.head())

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i]
        test = data.iloc[i:(i+step)]
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


predictions = backtest(sp500, model, predictors)

print("\n--- Backtest Prediction Counts ---")
print(predictions["Predictions"].value_counts())


backtest_precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"\nBacktest Precision (Initial Model): {backtest_precision:.4f}")

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:

    rolling_averages = sp500.rolling(horizon).mean()
    

    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
  
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]


sp500 = sp500.dropna()

improved_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

all_predictors = predictors + new_predictors

final_predictions = backtest(sp500, improved_model, all_predictors)

print("\n--- Final Prediction Counts (Improved Model) ---")
print(final_predictions["Predictions"].value_counts())

final_precision = precision_score(final_predictions["Target"], final_predictions["Predictions"])

print(f"\nFINAL PRECISION SCORE (Improved Model): {final_precision:.4f}")


print("\n--- Making a Prediction for AAPL ---")
aapl_ticker = yf.Ticker("AAPL")
aapl = aapl_ticker.history(period="max")

new_predictors_aapl = []
for horizon in horizons:
    rolling_averages = aapl.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    aapl[ratio_column] = aapl["Close"] / rolling_averages["Close"]

    aapl["Tomorrow"] = aapl["Close"].shift(-1)
    aapl["Target"] = (aapl["Tomorrow"] > aapl["Close"]).astype(int)
    
    trend_column = f"Trend_{horizon}"
    aapl[trend_column] = aapl.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors_aapl += [ratio_column, trend_column]

last_day_data = aapl.iloc[-1:][all_predictors].dropna()

if not last_day_data.empty:
    prediction_code = improved_model.predict(last_day_data)[0]
    prediction_proba = improved_model.predict_proba(last_day_data)[0][1] 

    if prediction_code == 1:
        print(f"The model predicts the price will go UP tomorrow.")
    else:
        print(f"The model predicts the price will go DOWN tomorrow.")
    
    print(f"Model Confidence (Probability of 'UP'): {prediction_proba * 100:.2f}%")
else:
    print("Could not make a prediction due to insufficient historical data for rolling averages.")



plot_data = sp500.join(final_predictions["Predictions"]).dropna()

# --- PLOT 1: TRADING SIGNALS ON PRICE CHART ---

# Create a new column for our "buy" signals.
# This column will have the 'Close' price on the day the model predicted UP (1),
# and 'NaN' otherwise. Matplotlib will not plot NaN values, so we will only
# see markers on the days we are interested in.
plot_data['buy_signal'] = np.where(plot_data['Predictions'] == 1, plot_data['Close'], np.nan)

# Create the plot
plt.figure(figsize=(16, 8))

# Plot the S&P 500 closing price
plt.plot(plot_data['Close'], label='S&P 500 Close Price', color='skyblue', alpha=0.6)

# Plot the "buy" signals as green triangles pointing up
plt.scatter(plot_data.index, plot_data['buy_signal'], label='Prediction: UP (Buy Signal)', marker='^', color='green', s=100, alpha=1.0)

# Add title and labels for clarity
plt.title('S&P 500 Price History with Model "UP" Predictions', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price USD ($)', fontsize=16)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)

# Display the plot. Note: This command will pause the script until you close the plot window.
print("\nDisplaying plot of trading signals. Close the plot window to finish the script.")
plt.show()


# --- PLOT 2: CUMULATIVE PERFORMANCE (A SIMPLIFIED EQUITY CURVE) ---

# This plot shows how many times the model was right vs. wrong over time.
# A consistently rising line means the model is performing well.

# We calculate the cumulative sum of correct predictions (Target == Predictions)
plot_data['correct_predictions'] = (plot_data['Target'] == plot_data['Predictions']).cumsum()
# We calculate the cumulative sum of total predictions made
plot_data['total_predictions'] = range(1, len(plot_data) + 1)
# We calculate the rolling accuracy
plot_data['rolling_accuracy'] = plot_data['correct_predictions'] / plot_data['total_predictions']

plt.figure(figsize=(16, 8))
plt.plot(plot_data['rolling_accuracy'], label='Model Cumulative Accuracy', color='orange')

# Add title and labels
plt.title('Cumulative Model Accuracy Over Time', fontsize=20)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)

plt.axhline(y=0.5, color='r', linestyle='--', label='50% Accuracy (Random Guess)')

print("\nDisplaying plot of cumulative accuracy. Close the plot window to finish the script.")
plt.show()
