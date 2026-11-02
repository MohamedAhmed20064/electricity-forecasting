from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.naive import naive_forecast

def sarima_forecast(history, horizon):
    try:
        y = history.iloc[-30:] # أحدث 30 سنة
        m = SARIMAX(y, order=(1, 1, 1)).fit(disp=False)
        return m.forecast(horizon).values
    except Exception:
        return naive_forecast(history, horizon)