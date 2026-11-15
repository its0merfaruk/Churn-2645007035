"""Proje ayarları. SEED burada tek yerde tutulur; tüm rastgele işlemler bunu kullanır."""
from pathlib import Path

# >>> BURAYI DOLDUR: öğrenci numaranın son 4 hanesi (örn. SEED = 1234) <<<
SEED = 7035

# Klasör yolları (bu dosyanın iki üst klasörü = proje kökü)
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
