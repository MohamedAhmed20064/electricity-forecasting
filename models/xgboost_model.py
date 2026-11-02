import pandas as pd
import numpy as np
import xgboost as xgb

FEATS = ["year", "lag_1", "lag_2", "lag_3", "roll_3"]

def create_xgb_features(s):
    f = pd.DataFrame(index=s.index)
    f["year"] = s.index.year if hasattr(s.index, 'year') else s.index
    f["lag_1"] = s.shift(1)
    f["lag_2"] = s.shift(2)
    f["lag_3"] = s.shift(3)
    f["roll_3"] = s.shift(1).rolling(3).mean()
    return f.assign(y=s).dropna()

class XGBForecaster:
    def __init__(self):
        self.m = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42)

    def fit(self, s, y=None):
       f = create_xgb_features(s)
       if len(f) > 0:
          self.m.fit(f[FEATS], f["y"])
       return self

    def predict(self, history, horizon):
        vals = list(history.iloc[-10:].values)
        last_year = history.index[-1].year if hasattr(history.index[-1], 'year') else int(history.index[-1])
        out = []
        
        for i in range(1, horizon + 1):
            t_year = last_year + i
            row = {
                "year": t_year,
                "lag_1": vals[-1],
                "lag_2": vals[-2],
                "lag_3": vals[-3],
                "roll_3": float(np.mean(vals[-3:]))
            }
            p = float(self.m.predict(pd.DataFrame([row])[FEATS])[0])
            out.append(p)
            vals.append(p)
        return np.array(out)