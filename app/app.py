"""منصة التنبؤ باستهلاك الكهرباء: مقارنة عدة دول وقارات.
التشغيل:  python clean_data.py   ثم   streamlit run app.py   (لا يحتاج train.py)"""
import io
import json
import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# يخلّي models/ و data/ يتلاقوا سواء app.py في المجلد الرئيسي أو جوه مجلد app/ (محلياً وعلى Streamlit Cloud)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = next((p for p in (_HERE, os.path.dirname(_HERE)) if os.path.isdir(os.path.join(p, "models"))), _HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
from models import lstm_model
from models import xgboost_model
from models import prophet
from models import sarima
st.set_page_config(page_title="Global Electricity Forecasting", page_icon="⚡", layout="wide")
st.markdown("""
<style>
/* 1. استدعاء الخط العربي (مثل خط Cairo أو Tajawal) من Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800;900&display=swap');
/* تطبيق الخط على التطبيق والواجهة والشريط الجانبي بالكامل */
html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Cairo', sans-serif !important;
}

/* 2. تغيير خلفية الواجهة الرئيسية (بدل السواد التام) لتدرج كحلي رمادي فاخر ومريح */
/* خلفية التطبيق بالكامل بالتدرج الجديد */
.stApp { 
    background: linear-gradient(90deg, #020024 0%, #090979 35%, #00D4FF 100%) !important;
}

/* خلفية القائمة الجانبية (Sidebar) بنفس التدرج رأسيًا */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020024 0%, #090979 50%, #00D4FF 100%) !important;
    border-right: 1px solid #00D4FF;
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] div {
    color: #ffffff !important;
    font-weight: 800 !important;
    text-shadow: 0px 1px 4px rgba(0, 0, 0, 0.8) !important;
}

/* 5. تحويل صناديق الاختيار (الداخل والإطار) للون الأبيض والخط أسود عريض */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
    border: 1px solid #0b2545 !important;
}

/* لون الخط والرموز داخل صناديق الاختيار */
[data-testid="stSidebar"] div[data-baseweb="select"] *,
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 700 !important;
}

/* خلفية الوسوم المحددة (Tags/Selected items) داخل مربع الاختيار */
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #e2e8f0 !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] * {
    color: #000000 !important;
}
/* 1. تصميم جديد لزرار أضف كل القارات */
[data-testid="stSidebar"] button {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00d4ff 100%) !important;
    border: 1px solid #00d4ff !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] button * {
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
}

[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
}

/* 2. تصميم احترافي لمربعات الاختيار خلفية كحلية زجاجية وإطار نيون */
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.85) !important;
    border: 1.5px solid #00d4ff !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] *,
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background-color: #090979 !important;
    border: 1px solid #00d4ff !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] * {
    color: #ffffff !important;
}

/* 6. إصلاح القوائم المنسدلة عند الفتح */
div[data-baseweb="popover"] *, 
div[data-baseweb="menu"] *, 
li[role="option"] * {
    color: #0f172a !important;
    background-color: #ffffff !important;
    font-weight: 700 !important;
}

li[role="option"]:hover {
    background-color: #57C785 !important;
    color: #ffffff !important;
}

/* 7. إصلاح صندوق التنبيهات والمعلومات (Info Alert) لتكون خلفيته فاتحة والكتابة بيضاء/سماوي زاهي */
.stAlert {
    background-color: rgba(30, 41, 59, 0.9) !important;
    border: 2px solid #38bdf8 !important;
    border-radius: 12px !important;
}

.stAlert p, .stAlert div {
    color: #38bdf8 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
}

/* 8. عناوين الواجهة الرئيسية والنصوص برة */
/* 8. عنوان منصة التنبؤ بتدرج ألوان نيون متداخلة جبارة */
.header-title { 
    background: linear-gradient(90deg, #00d4ff 0%, #57c785 50%, #eddd53 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2.8rem !important; 
    font-weight: 900 !important; 
    text-align: center !important; 
    margin-bottom: 8px !important;
    filter: drop-shadow(0px 2px 8px rgba(0, 212, 255, 0.4));
}

/* أزرار التشغيل والإيقاف الخاصة بالرسم التفاعلي */
.updatemenu-button,
.updatemenu-container button {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00d4ff 100%) !important;
    border: 1.5px solid #00d4ff !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-weight: 800 !important;
}

.header-subtitle { 
    color: #94a3b8 !important; 
    font-size: 1.1rem; 
    font-weight: 600; 
    text-align: center; 
    margin-bottom: 25px; 
}

/* 9. البطاقات الزجاجية والمجسمات في الشاشة الرئيسية */
.glass-card { 
    background: rgba(30, 41, 59, 0.7) !important; 
    border-radius: 16px; 
    padding: 22px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    margin-bottom: 20px; 
}
/* تعديل زرار "أضف كل القارات" وكل الأزرار داخل الشريط الجانبي */
[data-testid="stSidebar"] button {
    background-color: #ffffff !important;
    border: 2px solid #0b2545 !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

/* لون الكلمة والرمز جوه الزرار عشان تكون باينة جداً */
[data-testid="stSidebar"] button * {
    color: #0b2545 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
}

/* شكل الزرار لما تعدي عليه بالماوس (Hover) */
[data-testid="stSidebar"] button:hover {
    background-color: #0b2545 !important;
    border-color: #0b2545 !important;
}

[data-testid="stSidebar"] button:hover * {
    color: #ffffff !important;
}
/* إجبار أزرار Plotly التفاعلية على تغيير الخلفية */
.plotly .updatemenu-button rect,
.plotly .updatemenu-button path {
    fill: #1e3c72 !important;
    stroke: #00d4ff !important;
}

.plotly .updatemenu-button text {
    fill: #ffffff !important;
    font-weight: bold !important;
}
/* إصلاح تظليل أرقام الجداول باللون الأخضر النيون والخط الأسود العريض */
[data-testid="stDataFrame"] table td[style*="background"],
[data-testid="stDataFrame"] div[style*="background-color"],
[data-testid="stDataFrame"] [data-testimonial="data-grid-cell"] {
    background-color: #57c785 !important; /* أخضر نيون شفاف ورايق من بالتتك */
    color: #000000 !important;             /* خط أسود عريض عشان الرقم يوضح 100% */
    font-weight: 900 !important;
}

/* إجبار النص داخل الخلايا المظللة يكون أسود واضح */
[data-testid="stDataFrame"] td[style*="background"] * {
    color: #000000 !important;
    font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)
 
METHODS = {"ensemble": "المتوسط (Holt + اتجاه خطي)", "linear_drift": "اتجاه خطي", "ets_damped": "Holt Damped",
           "cagr": "نمو سنوي مركب CAGR", "naive": "Naive (آخر قيمة)"}
REGIONS = ["World", "Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
PALETTE = ["#0d47a1", "#e65100", "#00796b", "#8e24aa", "#c62828", "#5d4037", "#f9a825", "#00838f", "#6a1b9a", "#2e7d32"]
MAX_SEL, TEST_YEARS, CLEAN = 10, 5, "data/clean.csv"

# ---- موديلات مصر فقط ----
EGYPT = "Egypt"
EGY_MODELS = {"LSTM": "LSTM (شبكة عصبية)", "Prophet": "Prophet", "SARIMA": "SARIMA", "XGBoost": "XGBoost"}
EGY_AVG = "egy_avg"   # متوسط الأربع موديلات (خيار "عرض كل الموديلات")
EGY_AUTO = "auto"     # أفضل موديل لمصر تلقائياً حسب الـ rolling backtest
ROLL_K, ROLL_FOLDS = 3, 4   # 4 فترات اختبار × 3 سنين = آخر 12 سنة
EGY_UI = {"عرض كل الموديلات (All Models)": EGY_AVG, "🏅 الأفضل تلقائياً (Auto Best)": EGY_AUTO, "LSTM (Best Neural Network)": "LSTM",
          "Prophet (Best Trend)": "Prophet", "SARIMA": "SARIMA", "XGBoost": "XGBoost"}
LABELS = {**METHODS, **EGY_MODELS, EGY_AVG: "متوسط موديلات مصر الأربعة", EGY_AUTO: "الأفضل تلقائياً (مصر)"}
 
 
# ---------------------------------- بيانات ----------------------------------
def flag(iso):
    if not isinstance(iso, str) or len(iso) < 2:
        return "🌐"
    if iso.startswith("OWID"):
        return "🌍"
    return "".join(chr(127397 + ord(c)) for c in iso[:2].upper())
 
 
@st.cache_data(show_spinner="جاري تحميل البيانات...")
def load():
    if not os.path.exists(CLEAN):
        from data.clean_data import clean_data
        clean_data()
    d = pd.read_csv(CLEAN)
    d["year"] = d["year"].astype(int)
    return d
 
 
@st.cache_data(show_spinner=False)
def build(d):
    """{كيان: Series سنوية} + جدول الكيانات (دولة/قارة) — آخر فترة سنوات متصلة فقط."""
    ymax, series, meta = int(d["year"].max()), {}, {}
    for c, g in d.groupby("country"):
        s = pd.Series(g["load"].values.astype(float), index=g["year"].values).sort_index()
        s = s[s > 0]
        if s.empty:
            continue
        br = np.where(np.diff(s.index.values) > 1)[0]
        s = s.iloc[br[-1] + 1:] if len(br) else s
        if len(s) >= 11 and s.index[-1] >= ymax - 2:
            iso = g["iso_code"].dropna().iloc[0] if g["iso_code"].notna().any() else None
            series[c] = s.tail(30)
            meta[c] = {"iso": iso, "country": bool(iso) or c == "Kosovo"}
    return series, meta
 
 
# ---------------------------------- التنبؤ ----------------------------------
def _drift(y, h, k=10):
    t = y.iloc[-(k + 1):]
    return np.maximum(y.iloc[-1] + (t.iloc[-1] - t.iloc[0]) / (len(t) - 1) * np.arange(1, h + 1), 0)
 
 
def _holt(y, h):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        return np.asarray(ExponentialSmoothing(y.values, trend="add", damped_trend=True,
                                               initialization_method="estimated").fit().forecast(h))
    except Exception:
        return _drift(y, h)
 
 
EGY_CACHE = "data/egypt_forecasts.json"   # ملف الأرشيف: نتايج موديلات مصر محفوظة هنا


def _cache_load():
    try:
        with open(EGY_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(show_spinner="⏳ جاري تشغيل موديل مصر (أول مرة بس)...")
def egypt_forecast(y, h, name):
    """لو النتيجة محفوظة في الملف بتتحمل فوراً من غير تدريب، وإلا بتتحسب وتتحفظ."""
    key = f"{name}|{h}|{int(y.index[0])}-{int(y.index[-1])}|{len(y)}|{round(float(y.sum()), 4)}"
    store = _cache_load()
    if key in store:
        return np.asarray(store[key], float)
    out = _egypt_compute(y, h, name)
    store[key] = [float(v) for v in out]
    try:
        os.makedirs(os.path.dirname(EGY_CACHE), exist_ok=True)
        with open(EGY_CACHE, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except Exception:
        pass
    return out


def _egypt_compute(y, h, name):
    """يشغّل الموديل المدرَّب لمصر (fit على السلسلة الممررة ثم predict)."""
    y = y.astype(float)
    try:
        if name == "LSTM":
            try:
                from tensorflow import keras
                keras.utils.set_random_seed(42)
            except Exception:
                pass
            return np.asarray(lstm_model.LSTMForecaster(horizon=h, epochs=30).fit(y).predict(y, h, country_name=EGYPT), float)
        if name == "Prophet":
            return np.asarray(prophet.prophet_forecast(y, h), float)
        if name == "SARIMA":
            # SARIMA محتاج فهرس زمني (مش أرقام سنين عادية) وإلا بيرجع للـ naive بصمت
            ys = pd.Series(y.values, index=pd.date_range(f"{int(y.index[0])}-01-01", periods=len(y), freq="YS"))
            return np.asarray(sarima.sarima_forecast(ys, h), float)
        if name == "XGBoost":
            return np.asarray(xgboost_model.XGBForecaster().fit(y).predict(y, h), float)
    except Exception:
        pass
    return _drift(y, h)


def rolling_smape(y, m):
    """Rolling backtesting: متوسط sMAPE على ROLL_FOLDS فترات متتالية، كل فترة ROLL_K سنين."""
    errs = []
    for i in range(ROLL_FOLDS):
        end = len(y) - i * ROLL_K
        tr, te = y.iloc[:end - ROLL_K], y.iloc[end - ROLL_K:end]
        if len(tr) < 12:
            break
        errs.append(metrics(te, forecast(tr, ROLL_K, m))["sMAPE %"])
    return float(np.mean(errs)) if errs else None


@st.cache_data(show_spinner="⏳ جاري اختيار أفضل موديل لمصر...")
def best_egy(y):
    scores = {}
    for k in EGY_MODELS:
        try:
            v = rolling_smape(y, k)
            if v is not None:
                scores[k] = v
        except Exception:
            pass
    return min(scores, key=scores.get) if scores else list(EGY_MODELS)[0]


def forecast(y, h, m):
    if m == EGY_AUTO:
        return forecast(y, h, best_egy(y))
    if m in EGY_MODELS:
        return egypt_forecast(y, h, m)
    if m == EGY_AVG:
        return np.mean([egypt_forecast(y, h, k) for k in EGY_MODELS], axis=0)
    if m == "naive":
        return np.repeat(y.iloc[-1], h)
    if m == "linear_drift":
        return _drift(y, h)
    if m == "cagr":
        t = y.iloc[-6:]
        g = np.clip((t.iloc[-1] / t.iloc[0]) ** (1 / (len(t) - 1)), 0.9, 1.1)
        return y.iloc[-1] * g ** np.arange(1, h + 1)
    if m == "ets_damped":
        return _holt(y, h)
    return (_drift(y, h) + _holt(y, h)) / 2
 
 
def holdout(y, m, k=TEST_YEARS):
    """نخفي آخر k سنوات، ونتنبأ بها من الباقي، ثم نقارن بالفعلي."""
    if len(y) < k + 8:
        return None
    return y.iloc[-k:], pd.Series(forecast(y.iloc[:-k], k, m), index=y.index[-k:])
 
 
def metrics(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    e, den = a - p, np.abs(a) + np.abs(p)
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "sMAPE %": 100 * np.mean(np.where(den == 0, 0, 2 * np.abs(e) / np.where(den == 0, 1, den)))}
 
 
def norm(s):
    return 100 * s / s.iloc[0]
 
 
def style(fig, title, yt, log=False):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(color="#00D4FF", size=18)),
        paper_bgcolor="rgba(2, 0, 36, 0.6)", 
        plot_bgcolor="rgba(9, 9, 121, 0.4)",
        hovermode="x unified",
        xaxis=dict(
            title=dict(text="السنة / الدولة", font=dict(color="#ffffff", size=14, weight="bold")),
            tickfont=dict(color="#ffffff", size=12, weight="bold"),
            gridcolor="rgba(255,255,255,0.1)"
        ),
        yaxis=dict(
            title=dict(text=yt, font=dict(color="#ffffff", size=14, weight="bold")),
            tickfont=dict(color="#ffffff", size=12, weight="bold"),
            type="log" if log else "linear",
            gridcolor="rgba(255,255,255,0.1)"
        ),
        legend=dict(font=dict(color="#ffffff", size=12, weight="bold")),
        
        # 🎨 التنسيق المباشر لأزرار التشغيل والإيقاف المدمجة في Plotly
        updatemenus=[dict(
            type="buttons",
            showactive=True,
            bgcolor="#1e3c72",       # خلفية الأزرق الملكي
            bordercolor="#00d4ff",   # إطار سماوي نيون
            borderwidth=1.5,
            font=dict(color="#ffffff", size=13, family="Changa", weight="bold"), # خط أبيض عريض
            pad=dict(r=10, t=10),
            x=0.1,
            y=0
        )]
    )
    return fig
 
 
def card(on):
    st.markdown("<div class='glass-card'>" if on else "</div>", unsafe_allow_html=True)
 
 
# ---------------------------------- الواجهة ----------------------------------
try:
    data = load()
    series, meta = build(data)
except Exception as e:
    st.error(f"❌ تعذر تحميل البيانات: {e}\nضع owid_energy.csv داخل مجلد data ثم شغّل clean_data.py.")
    st.stop()
 
st.markdown("<div class='header-title'>⚡ منصة التنبؤ الذكي باستهلاك الكهرباء</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>اختر عدة دول أو قارات وقارن بينها: التاريخ، الدقة، والتنبؤ المستقبلي</div>",
            unsafe_allow_html=True)
 
with st.sidebar:
    st.header("🎛️ لوحة التحكم")
    all_mode = st.toggle("🌍 كل الدول (سباق متحرك)", value=False, help="ترتيب كل الدول عبر السنوات حتى 2030")
    kind = st.radio("نوع الكيانات", ["الكل", "دول فقط", "قارات ومناطق فقط"], horizontal=True)
    opts = [c for c in series if kind == "الكل" or (kind == "دول فقط") == meta[c]["country"]]
    opts = [c for c in REGIONS if c in opts] + sorted(c for c in opts if c not in REGIONS)
    if st.button("🌍 أضف كل القارات"):
        st.session_state["sel"] = [c for c in REGIONS[1:] if c in series]
    if "sel" not in st.session_state:
          st.session_state["sel"] = []
    st.session_state["sel"] = [c for c in st.session_state["sel"] if c in opts]
    selected = st.multiselect(f"اختر الدول / القارات (حتى {MAX_SEL}):", opts, key="sel", max_selections=MAX_SEL,
                              format_func=lambda c: f"{flag(meta[c]['iso']) if c not in REGIONS else '🌍'} {c}")

# عناصر التحكم في الصفحة الرئيسية (زي الشكل الأصلي بالظبط)
has_egy = EGYPT in selected
others = [c for c in selected if c != EGYPT]
egy_key, method = EGY_AVG, list(METHODS)[0]

if has_egy:
    st.success("🎯 **Egypt Tuned Pipeline Active:** تم تفعيل موديلات التنبؤ المعالجة لمصر.")
    egy_key = EGY_UI[st.selectbox("🔮 اختر موديل التنبؤ المخصص لمصر:", list(EGY_UI))]
    if egy_key == EGY_AUTO and EGYPT in series:
        st.caption(f"🏅 الموديل المختار تلقائياً لمصر: {EGY_MODELS[best_egy(series[EGYPT])]}")
# الطرق البسيطة السريعة: لباقي الدول (أو لو مفيش مصر)
if others or not has_egy:
    method = st.selectbox("🤖 طريقة التنبؤ" + (" (باقي الدول)" if has_egy else ""), list(METHODS),
                          format_func=METHODS.get)
normed = st.checkbox("📈 مؤشر (100 = أول سنة) لمقارنة الأحجام المختلفة")
logy = st.checkbox("مقياس لوغاريتمي", value=False)
show_out = st.checkbox("إظهار القيم الشاذة (IQR)", value=False)
H = st.slider("🔮 مدى التنبؤ (سنوات)", 1, 10, 5)


def method_for(c):
    """مصر -> موديلها المختار، أي دولة تانية -> الطريقة البسيطة."""
    return egy_key if c == EGYPT else method


G = max(int(s.index[-1]) for s in series.values())


@st.cache_data(show_spinner="⏳ جاري حساب توقعات كل الدول...")
def all_forecasts(m, egy, h_years=5):
    res = {}
    for c, s in series.items():
        last = int(s.index[-1])
        res[c] = pd.Series(forecast(s, h_years, egy if c == EGYPT else m), index=np.arange(last + 1, last + 1 + h_years))
    return res


if all_mode:
    names = [c for c in series if meta[c]["country"]]
    fa = all_forecasts(method, egy_key, h_years=H)
 
    def val(c, y):
        s = series[c]
        if y < s.index[0]:
            return 0.0
        if y <= s.index[-1]:
            return float(s.loc[y])
        return float(fa[c].iloc[min(y - int(s.index[-1]), len(fa[c])) - 1])
 
    yrs, TOP = list(range(2000, 2031)), 12
    colr = {c: f"hsl({i * 47 % 360},75%,50%)" for i, c in enumerate(names)}
    frames, xmax = [], 1.0
    for y in yrs:
        top = sorted(((val(c, y), c) for c in names), reverse=True)[:TOP][::-1]
        xmax = max(xmax, top[-1][0])
        lab = [f"{flag(meta[c]['iso'])} {c}" for _, c in top]
        frames.append(go.Frame(name=str(y), layout=dict(title_text=f"<b>أكبر {TOP} دولة في استهلاك الكهرباء — {y}"
                                                                    + (" (تنبؤ)" if y > G else "") + "</b>"),
                               data=[go.Bar(x=[v for v, _ in top], y=lab, orientation="h",
                                            marker_color=[colr[c] for _, c in top],
                                            text=[f"{v:,.0f}" for v, _ in top], textposition="outside")]))
    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        title=dict(text=frames[0].layout.title.text, font=dict(color="#0d47a1", size=20)), height=640,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.7)", showlegend=False,
        xaxis=dict(range=[0, xmax * 1.18], title="TWh"), yaxis=dict(tickfont=dict(size=14)),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.12, buttons=[
            dict(label="▶ تشغيل", method="animate", args=[None, {"frame": {"duration": 650, "redraw": True},
                                                               "fromcurrent": True, "transition": {"duration": 350}}]),
            dict(label="⏸ إيقاف", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])])],
        sliders=[dict(x=0.12, len=0.88, y=-0.1, currentvalue=dict(prefix="السنة: "),
                      steps=[dict(method="animate", label=str(y) + ("*" if y > G else ""),
                                  args=[[str(y)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                                   "transition": {"duration": 300}}]) for y in yrs])])
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"* سنوات بعد {G} هي تنبؤ بطريقة «{METHODS[method]}». اضغط ▶ أو حرّك شريط السنوات.")
 
    rows = []
    for c in names:
        s, f = series[c], fa[c]
        rows.append({"الدولة": f"{flag(meta[c]['iso'])} {c}", f"آخر قيمة ({int(s.index[-1])})": s.iloc[-1],
                     "المتوقع 2030": f.iloc[-1], "التغير %": 100 * (f.iloc[-1] / s.iloc[-1] - 1),
                     "نمو سنوي %": 100 * ((f.iloc[-1] / s.iloc[-1]) ** (1 / len(f)) - 1)})
    allt = pd.DataFrame(rows).set_index("الدولة")
    a, b = st.columns(2)
    a.subheader("🚀 الأسرع نمواً حتى 2030")
    a.dataframe(allt.sort_values("نمو سنوي %", ascending=False).head(10).round(2), use_container_width=True)
    b.subheader("🏆 الأعلى استهلاكاً في 2030")
    b.dataframe(allt.sort_values("المتوقع 2030", ascending=False).head(10).round(2), use_container_width=True)
    st.subheader("📋 كل الدول")
    st.dataframe(allt.round(2), use_container_width=True)
    st.download_button("📄 تصدير كل الدول CSV", allt.to_csv().encode("utf-8-sig"), "all_countries_forecast.csv", "text/csv")
    st.stop()
 
if not selected:
    st.info("👈 اختر دولة أو قارة واحدة على الأقل من الشريط الجانبي.")
    st.stop()
 
col = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(selected)}
method_name = " + ".join(dict.fromkeys(LABELS[method_for(c)] for c in selected))
unit = "المؤشر (100 = أول سنة)" if normed else "الاستهلاك (TWh)"
fut = {c: pd.Series(forecast(series[c], H, method_for(c)), index=series[c].index[-1] + np.arange(1, H + 1)) for c in selected}
if len(selected) <= 4:
    for cc, c in zip(st.columns(len(selected)), selected):
        s, f = series[c], fut[c]
        with cc.container(border=True):
            st.markdown(f"### {flag(meta[c]['iso']) if c not in REGIONS else '🌍'} {c}")
            st.metric(f"آخر قيمة فعلية ({int(s.index[-1])})", f"{s.iloc[-1]:,.1f} TWh")
            st.metric(f"المتوقع ({int(f.index[-1])})", f"{f.iloc[-1]:,.1f} TWh", f"{100 * (f.iloc[-1] / s.iloc[-1] - 1):+.1f}%")
            st.metric("النمو السنوي المتوقع", f"{100 * ((f.iloc[-1] / s.iloc[-1]) ** (1 / H) - 1):+.2f}%")
 
t1, t2, t3, t4, t5 = st.tabs(["📊 الاستهلاك التاريخي", "🎯 الفعلي مقابل المتوقع", "🏆 مقارنة الطرق",
                              "🔮 التوقعات المستقبلية", "📥 التصدير"])
 
with t1:
    card(True)
    fig = go.Figure()
    for c in selected:
        s = series[c]
        fig.add_trace(go.Scatter(x=s.index, y=(norm(s) if normed else s).values, name=c, mode="lines",
                                 line=dict(color=col[c], width=3)))
        if show_out:
            o = data[(data["country"] == c) & data["is_outlier"].astype(bool) & data["year"].isin(s.index)]
            if len(o):
                oy = o["load"].values / (s.iloc[0] if normed else 1) * (100 if normed else 1)
                fig.add_trace(go.Scatter(x=o["year"], y=oy, mode="markers", name=f"شاذة: {c}",
                                         marker=dict(color="#b71c1c", size=9, symbol="circle-open", line=dict(width=2))))
    st.plotly_chart(style(fig, "سلسلة الاستهلاك التاريخي", unit, logy), use_container_width=True)
    rows = [{"الكيان": c, "من": int(s.index[0]), "إلى": int(s.index[-1]), "أول قيمة": s.iloc[0], "آخر قيمة": s.iloc[-1],
             "أعلى قيمة": s.max(), "نمو سنوي مركب %": 100 * ((s.iloc[-1] / s.iloc[0]) ** (1 / (len(s) - 1)) - 1)}
            for c, s in ((c, series[c]) for c in selected)]
    st.subheader("📋 ملخص المقارنة (TWh)")
    st.dataframe(pd.DataFrame(rows).set_index("الكيان").round(2), use_container_width=True)
    card(False)
 
hold = {c: holdout(series[c], method_for(c)) for c in selected}

with t2:
    card(True)
    st.caption(f"نخفي آخر {TEST_YEARS} سنوات من كل كيان ونتنبأ بها بطريقة «{method_name}» ثم نقارن بالفلي.")
    fig, rows = go.Figure(), []
    for c in selected:
        if hold[c] is None:
            continue
        a, p = hold[c]
        base = a.iloc[0] if normed else 1
        k = 100 / base if normed else 1
        if len(selected) == 1:
            sg = 1.645 * np.std(a.values - p.values)
            fig.add_trace(go.Scatter(x=p.index, y=(p.values + sg) * k, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=p.index, y=(p.values - sg) * k, fill="tonexty", line=dict(width=0),
                                     name="مجال ±1.645σ", fillcolor="rgba(13,71,161,0.15)"))
        fig.add_trace(go.Scatter(x=a.index, y=a.values * k, name=f"{c} (فعلي)", line=dict(color=col[c], width=2.6)))
        fig.add_trace(go.Scatter(x=p.index, y=p.values * k, name=f"{c} (متوقع)",
                                 line=dict(color=col[c], width=2.6, dash="dash")))
        if c == EGYPT and egy_key == EGY_AVG:
            for mk in EGY_MODELS:
                h2 = holdout(series[c], mk)
                if h2 is not None:
                    fig.add_trace(go.Scatter(x=h2[1].index, y=h2[1].values * k, name=f"مصر - {mk}",
                                             line=dict(width=1.8, dash="dot")))
        m, b = metrics(a, p), metrics(a, np.repeat(series[c].iloc[-TEST_YEARS - 1], TEST_YEARS))
        rows.append({"الكيان": c, **m, "التحسن عن Naive %": 100 * (1 - m["MAE"] / b["MAE"]) if b["MAE"] else 0.0})
    # استخدام method_name الآمنة لتفادي KeyError مع موديلات مصر
    st.plotly_chart(style(fig, f"الدقة - تنبؤ {method_name}", unit), use_container_width=True)
    if rows:
        st.subheader("📋 مقاييس الدقة لكل كيان")
        st.dataframe(pd.DataFrame(rows).set_index("الكيان").round(2), use_container_width=True)
    card(False)
 
with t3:
    card(True)
    st.subheader("🏆 sMAPE % لكل طريقة (الأقل أفضل)")
    mat = {}
    for c in selected:
        if hold[c] is not None:
            c_res = {}
            for m in list(METHODS) + (list(EGY_MODELS) + [EGY_AVG] if c == EGYPT else []):
                try:
                    c_res[LABELS[m]] = metrics(*holdout(series[c], m))["sMAPE %"]
                except Exception:
                    pass
            if c_res:
                mat[c] = c_res
    if mat:
        df_m = pd.DataFrame(mat)
        df_m["المتوسط"] = df_m.mean(axis=1)
        st.dataframe(df_m.round(2).style.highlight_min(axis=0, props='background-color: #57c785; color: #000000; font-weight: 900;'), use_container_width=True)
        st.caption("الأخضر = أفضل طريقة لكل كيان. اختبار على آخر 5 سنوات فقط (تشمل تعافي ما بعد كورونا).")
    st.subheader(f"🔁 Rolling Backtesting: متوسط sMAPE % على {ROLL_FOLDS} فترات × {ROLL_K} سنين")
    roll = {}
    for c in selected:
        r = {}
        for m in list(METHODS) + (list(EGY_MODELS) + [EGY_AVG] if c == EGYPT else []):
            try:
                v = rolling_smape(series[c], m)
                if v is not None:
                    r[LABELS[m]] = v
            except Exception:
                pass
        if r:
            roll[c] = r
    if roll:
        st.dataframe(pd.DataFrame(roll).round(2).style.highlight_min(
            axis=0, props='background-color: #57c785; color: #000000; font-weight: 900;'), use_container_width=True)
        for c, r in roll.items():
            best = min(r, key=r.get)
            st.info(f"🏅 أفضل طريقة لـ {c}: {best} (متوسط sMAPE = {r[best]:.2f}%)")
        if EGYPT in roll:
            r = roll[EGYPT]
            fast = {LABELS[m]: r[LABELS[m]] for m in METHODS if LABELS[m] in r}
            hard = {LABELS[m]: r[LABELS[m]] for m in EGY_MODELS if LABELS[m] in r}
            if fast and hard:
                bf, bh = min(fast, key=fast.get), min(hard, key=hard.get)
                if r[bh] < r[bf]:
                    st.info(f"✅ في مصر الموديلات الصعبة كسبت: {bh} ({r[bh]:.2f}%) أدق من أفضل طريقة سريعة {bf} ({r[bf]:.2f}%).")
                else:
                    st.info(f"ℹ️ في مصر أفضل طريقة سريعة {bf} ({r[bf]:.2f}%) كانت أدق من أفضل موديل صعب {bh} ({r[bh]:.2f}%).")
        st.caption("كل فترة اختبار بتتدرب على اللي قبلها بس (من غير ما تشوف المستقبل)، والرقم هو المتوسط عبر الفترات.")
    card(False)
 
with t4:
    card(True)
    fig = go.Figure()
    for c in selected:
        s, f = series[c].iloc[-15:], fut[c]
        base = s.iloc[0] if normed else 1
        k = 100 / base if normed else 1
        if len(selected) == 1 and hold[c] is not None:
            sg = np.sqrt(np.mean(np.log(hold[c][0] / hold[c][1]) ** 2))
            w = np.exp(1.645 * sg * np.sqrt(np.arange(1, H + 1)))
            fig.add_trace(go.Scatter(x=f.index, y=f.values * w * k, line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=f.index, y=f.values / w * k, fill="tonexty", line=dict(width=0),
                                     name="مجال تقريبي 90%", fillcolor="rgba(13,71,161,0.15)"))
        fig.add_trace(go.Scatter(x=s.index, y=s.values * k, name=f"{c} (فعلي)", line=dict(color=col[c], width=3)))
        fig.add_trace(go.Scatter(x=[s.index[-1], *f.index], y=[s.iloc[-1] * k, *(f.values * k)], name=f"{c} (توقع)",
                                 line=dict(color=col[c], width=3, dash="dash")))
        if c == EGYPT and egy_key == EGY_AVG:
            for mk in EGY_MODELS:
                fm = forecast(series[c], H, mk)
                fig.add_trace(go.Scatter(x=[s.index[-1], *f.index], y=[s.iloc[-1] * k, *(fm * k)], name=f"مصر - {mk}",
                                         line=dict(width=1.8, dash="dot")))
    st.plotly_chart(style(fig, f"التنبؤ للسنوات القادمة - {method_name}", unit, logy), use_container_width=True)
 
    tbl = pd.DataFrame(fut)
    tbl.index.name = "السنة"
    st.subheader("📋 قيم التنبؤ (TWh)")
    st.dataframe(tbl.round(2), use_container_width=True)
    chg = pd.DataFrame({"آخر قيمة فعلية": {c: series[c].iloc[-1] for c in selected},
                        f"المتوقع بعد {H} سنة": {c: fut[c].iloc[-1] for c in selected}})
    chg["التغير %"] = 100 * (chg.iloc[:, 1] / chg.iloc[:, 0] - 1)
    chg["نمو سنوي %"] = 100 * ((chg.iloc[:, 1] / chg.iloc[:, 0]) ** (1 / H) - 1)
    bar = go.Figure(go.Bar(x=chg.index, y=chg["التغير %"], marker_color=[col[c] for c in chg.index],
                           text=chg["التغير %"].round(1).astype(str) + "%", textposition="outside"))
    st.plotly_chart(style(bar, f"نسبة التغير المتوقعة بعد {H} سنوات", "%").update_layout(xaxis_title="", hovermode="closest"),
                    use_container_width=True)
    st.dataframe(chg.round(2), use_container_width=True)
    st.caption("⚠️ التنبؤ امتداد إحصائي للاتجاه التاريخي، ولا يتضمن سياسات أو صدمات مستقبلية. الدقة تقل مع طول المدى.")
    card(False)
 
with t5:
    card(True)
    st.subheader("📥 تصدير النتائج (للكيانات المختارة)")
    hist = pd.concat({c: series[c] for c in selected}, names=["country", "year"]).rename("TWh").reset_index()
    fut_df = pd.concat({c: fut[c] for c in selected}, names=["country", "year"]).rename("forecast_TWh").reset_index()
    ho = pd.concat({c: pd.DataFrame({"actual": hold[c][0], "predicted": hold[c][1]}) for c in selected
                    if hold[c] is not None}, names=["country", "year"]).reset_index() if any(hold.values()) else pd.DataFrame()
    met = pd.DataFrame(mat).T.rename_axis("country").reset_index() if mat else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    for cc, lab, df_, fn in [(c1, "📄 التاريخي CSV", hist, "historical"), (c2, "📄 التوقعات CSV", fut_df, "forecast"),
                             (c3, "📄 الفعلي/المتوقع CSV", ho, "holdout"), (c4, "📄 مقارنة الطرق CSV", met, "methods_smape")]:
        cc.download_button(lab, df_.to_csv(index=False).encode("utf-8-sig"), f"{fn}.csv", "text/csv",
                           use_container_width=True, disabled=df_.empty)
    try:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            for n, df_ in [("Historical", hist), ("Forecast", fut_df), ("Holdout", ho), ("Methods_sMAPE", met)]:
                if not df_.empty:
                    df_.to_excel(w, sheet_name=n, index=False)
        st.download_button("📊 تصدير كل شيء Excel", buf.getvalue(), "electricity_comparison.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    except ImportError:
        st.caption("ثبّت openpyxl لتصدير Excel:  pip install openpyxl")
    card(False)