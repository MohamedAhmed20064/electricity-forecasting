import numpy as np

def naive_forecast(history, horizon):
    """تنبؤ Naive بسيط"""
    return np.repeat(history.iloc[-1], horizon)

def seasonal_naive_forecast(history, horizon, season=1):
    """تنبؤ Seasonal Naive"""
    if len(history) >= season:
        return history.iloc[-season:-season + horizon].values if horizon < season else np.resize(history.iloc[-season:].values, horizon)
    return naive_forecast(history, horizon)