"""شغّله مرة واحدة:  python app/prefill_cache.py   (من مجلد المشروع الرئيسي)
بيجرّب كل موديلات مصر × كل مدى تنبؤ (1..10) عشان ملف data/egypt_forecasts.json يتملي بالكامل.
بعدها الموديلات بتفتح فوراً من غير تدريب."""
import os
import sys
 
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
# مجلد المشروع الرئيسي = اللي فيه models/ (سواء app.py جنبه أو جوه مجلد app/)
ROOT = next((p for p in (HERE, PARENT) if os.path.isdir(os.path.join(p, "models"))), PARENT)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
 
from streamlit.testing.v1 import AppTest
 
at = AppTest.from_file(os.path.join(HERE, "app.py"), default_timeout=3600)
at.run()
if at.exception:
    print("❌ التطبيق فيه خطأ:", at.exception[0].value)
    sys.exit(1)
at.multiselect(key="sel").set_value(["Egypt"]).run()
options = list(at.selectbox[0].options)
for opt in options:
    for h in range(1, 11):
        at.selectbox[0].set_value(opt)
        at.slider[0].set_value(h)
        at.run()
        print(f"{opt}  |  H={h}  |  errors={len(at.exception)}", flush=True)
print("خلص ✅ — الملف data/egypt_forecasts.json جاهز")
 