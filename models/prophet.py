import pandas as pd
from prophet import Prophet
from models.naive import naive_forecast

def prophet_forecast(history, horizon):
    try:
        d = history.reset_index()
        d.columns = ["ds", "y"]
        if not pd.api.types.is_datetime64_any_dtype(d['ds']):
            d['ds'] = pd.to_datetime(d['ds'].astype(str) + '-01-01')
            
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(d)
        fut = m.make_future_dataframe(periods=horizon, freq="YS")
        preds = m.predict(fut)
        return preds["yhat"].iloc[-horizon:].values
    except Exception:
        return naive_forecast(history, horizon)