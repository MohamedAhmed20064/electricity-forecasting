# ⚡ منصة التنبؤ الذكي باستهلاك الكهرباء

مقارنة استهلاك الكهرباء لعدة دول وقارات (بيانات Our World in Data) مع تنبؤ مستقبلي.

## الفكرة
- **باقي الدول والقارات:** 5 طرق سريعة (المتوسط، اتجاه خطي، Holt Damped، CAGR، Naive).
- **مصر فقط:** 4 موديلات مدرَّبة (LSTM, Prophet, SARIMA, XGBoost) + متوسطهم + اختيار «الأفضل تلقائياً».
- **التقييم:** holdout على آخر 5 سنين + **Rolling Backtesting** (4 فترات × 3 سنين) عشان المقارنة تبقى عادلة.

## التشغيل محلياً
```bash
pip install -r requirements.txt
pip install tensorflow-cpu      # لو عايز تدرّب LSTM
python data/clean_data.py
streamlit run app.py
```
أول مرة تختار موديل لمصر بيتدرّب ويتحفظ في `data/egypt_forecasts.json`، وبعدها بيتحمل فوراً.
لو عدّلت في أي ملف موديل أو في البيانات: امسح `data/egypt_forecasts.json`.

## تجهيز الملف قبل النشر (مرة واحدة)
```bash
python prefill_cache.py
```

## النشر على Streamlit Cloud
1. ارفع المشروع على GitHub (فيه `app.py` و`models/` و`data/clean.csv` و`data/egypt_forecasts.json` و`requirements.txt`).
2. على share.streamlit.io: New app ← اختار الريبو ← Main file: `app.py`.
3. خد الرابط وحطه هنا.

## هيكل المشروع
```
app.py                  الواجهة
models/                 lstm_model.py, xgboost_model.py, prophet.py, sarima.py, naive.py
data/                   clean_data.py, clean.csv, egypt_forecasts.json
prefill_cache.py        يملي ملف النتايج مرة واحدة
```
