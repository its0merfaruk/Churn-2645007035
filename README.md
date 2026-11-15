# Veri Madenciliği Dönem Projesi – Proje 2: Müşteri Kaybı (Churn) Tahmini

- Öğrenci: <ad soyad> – <numara>
- SEED: <numaranın son 4 hanesi> (`src/config.py` içinde)

## Veri seti
Telco Customer Churn (Kaggle, IBM örnek verisi). Ham veri depoda YOKTUR.
1. Kaggle'da "Telco Customer Churn" sayfasından zip'i indir (ücretsiz hesap gerekir).
2. İçindeki `WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyasını `data/` klasörüne koy.
3. Atıf (citation) bilgisi Kaggle sayfasındadır; kaynakçaya eklenecek.

## Kurulum
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Çalıştırma (proje kök klasöründen)
```bash
python notebooks/01_eda.py       # grafikler results/ klasörüne kaydedilir
python notebooks/02_baseline.py  # baseline tablosu results/baseline_cv.csv
```

## Klasörler
- `src/`: ortak fonksiyonlar (SEED, veri yükleme)
- `notebooks/`: EDA, ön işleme, modelleme
- `results/`: tablolar ve grafikler
- `report/`: rapor PDF'i
- `JOURNAL.md`: haftalık çalışma günlüğü
