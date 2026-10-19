import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def train_power_model(df):

    data = df.copy()

    # Create time/order feature
    data["Time_Index"] = np.arange(len(data))

    # Previous power value
    data["Previous_Power"] = data["Total_Power"].shift(1)

    # Remove first row because it has no previous value
    data = data.dropna(
        subset=["Previous_Power", "Total_Power"]
    )

    X = data[
        [
            "Time_Index",
            "Previous_Power"
        ]
    ]

    y = data["Total_Power"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Test model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return model, mae, r2


def predict_next_power(model, df):

    last_power = float(
        df["Total_Power"].iloc[-1]
    )

    next_index = len(df)

    input_data = pd.DataFrame(
        {
            "Time_Index": [next_index],
            "Previous_Power": [last_power]
        }
    )

    prediction = model.predict(
        input_data
    )[0]

    return prediction


def get_status(predicted_power, average_power):

    difference = (
        abs(predicted_power - average_power)
        / average_power
    ) * 100

    if difference <= 10:

        return "NORMAL"

    elif difference <= 20:

        return "WARNING"

    else:

        return "CRITICAL"