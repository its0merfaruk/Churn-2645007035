"""Veriyi okuma ve temel temizlik. Hem EDA hem modelleme aynı fonksiyonu kullanır."""
import pandas as pd

from src.config import SEED, DATA_PATH


def load_data():
    """CSV'yi okur, ödevdeki kurala göre SEED ile %80 örnekler ve temizler."""
    if SEED is None:
        raise ValueError("src/config.py içindeki SEED değerini öğrenci numaranın son 4 hanesiyle doldur.")

    df = pd.read_csv(DATA_PATH)

    # Ödev kuralı: veri önce SEED ile %80 örneklenir
    df = df.sample(frac=0.8, random_state=SEED)

    # customerID sadece kimlik numarası, tahmine katkısı yok -> at
    df = df.drop(columns=["customerID"])

    # TotalCharges bazı satırlarda boş metin (" ") olduğu için sayı sanılmıyor.
    # errors="coerce": sayıya çevrilemeyenleri NaN (kayıp veri) yapar.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Hedef: Yes -> 1 (ayrıldı), No -> 0 (kaldı)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df.reset_index(drop=True)


def split_xy(df):
    """Tabloyu girdi (X) ve hedef (y) olarak ayırır."""
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y
