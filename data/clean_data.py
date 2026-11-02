import os
import pandas as pd
import numpy as np

def clean_data():
    raw_path = "owid_energy.csv" if os.path.exists("owid_energy.csv") else "data/owid_energy.csv"
    if not os.path.exists(raw_path):
        print("❌ لم يتم العثور على ملف البيانات الخام owid_energy.csv")
        return

    print("⏳ [1/2] جاري قراءة وتنظيف البيانات لكل دول العالم...")
    df = pd.read_csv(raw_path)

    # استخراج عمود الاستهلاك الموحد (TWh)
    df['load_raw'] = df['electricity_demand_twh'].fillna(df['electricity_generation_twh'])

    # اختيار الأعمدة والتصفية
    clean_df = df[['country', 'year', 'iso_code', 'population', 'gdp', 'load_raw']].dropna(subset=['load_raw']).copy()

    # الاستكمال الزمني لكل دولة/قارة
    clean_df['load'] = clean_df.groupby('country')['load_raw'].transform(
        lambda g: g.interpolate(method='linear').bfill().ffill()
    )

    # كشف القيم الشاذة
    def check_outliers(group):
        q1, q3 = group['load'].quantile(0.25), group['load'].quantile(0.75)
        iqr = q3 - q1
        return (group['load'] < (q1 - 1.5 * iqr)) | (group['load'] > (q3 + 1.5 * iqr))

    clean_df['is_outlier'] = clean_df.groupby('country', group_keys=False).apply(check_outliers)
    clean_df['timestamp'] = pd.to_datetime(clean_df['year'].astype(str) + '-01-01')
    clean_df = clean_df.sort_values(['country', 'year']).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    clean_df.to_csv("data/clean.csv", index=False)
    print(f"✅ [2/2] تم إنشاء ملف data/clean.csv لـ {clean_df['country'].nunique()} دولة/قارة بنجاح!")

if __name__ == "__main__":
    clean_data()