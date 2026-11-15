import sys
sys.path.append(".")                     # Python, proje klasörünü de arasın (src'yi bulabilsin)

import pandas as pd                      # tablolarla çalışan kütüphane
from src.config import SEED, DATA_PATH   # config.py'deki ayarları çağır

import matplotlib.pyplot as plt          # grafik çizme kütüphanesi

import seaborn as sns                    # güzel grafikler (ısı haritası için)

# CSV'yi okuyup "df" adlı tabloya koy
df = pd.read_csv(DATA_PATH)

print("SEED:", SEED)
print("Boyut (satır, sütun):", df.shape)


# Ödev kuralı: tablonun rastgele %80'ini al.
# random_state=SEED sayesinde her çalıştırışta AYNI satırlar gelir.
df = df.sample(frac=0.8, random_state=SEED)
print("Örneklemden sonra:", df.shape)

# customerID sadece kimlik numarası, tahmine yardım etmez -> at
df = df.drop(columns=["customerID"])

# Her sütunun tipine ve dolu hücre sayısına bak
df.info()

# TotalCharges'ı sayıya çevir; çevrilemeyenler NaN (boş) olsun
sayi = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Çevrilemeyen satır sayısı:", sayi.isna().sum())

# Hangi müşterilerde sorun var?
print(df.loc[sayi.isna(), ["tenure", "MonthlyCharges", "TotalCharges"]])

# Çevrilmiş sütunu tabloya geri koy
df["TotalCharges"] = sayi

# Hedef: Yes -> 1 (ayrıldı), No -> 0 (kaldı)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(df["Churn"].value_counts())

# GRAFİK 1: kalan ve ayrılan müşteri sayısı
sayilar = df["Churn"].value_counts().sort_index()      # sırayla 0 ve 1
sayilar.index = ["Kalan", "Ayrılan"]                   # etiketleri okunur yap
sayilar.plot(kind="bar", color=["tab:blue", "tab:red"])
plt.title("Hedef değişken dağılımı")
plt.ylabel("Müşteri sayısı")
plt.xticks(rotation=0)                                 # yazılar yatay dursun
plt.tight_layout()
plt.savefig("results/g1_churn_dagilimi.png", dpi=150)  # dosyaya kaydet
plt.show()                                             # ekranda göster

# GRAFİK 2: tenure ve MonthlyCharges dağılımı (kalan / ayrılan ayrı renkte)
fig, eksenler = plt.subplots(1, 2, figsize=(10, 4))    # yan yana 2 kutu
for ax, sutun in zip(eksenler, ["tenure", "MonthlyCharges"]):
    for deger, ad in [(0, "Kalan"), (1, "Ayrılan")]:
        ax.hist(df.loc[df["Churn"] == deger, sutun], bins=30, alpha=0.6, label=ad)
    ax.set_title(sutun)
    ax.legend()
plt.tight_layout()
plt.savefig("results/g2_histogramlar.png", dpi=150)
plt.show()

sayisal = ["tenure", "MonthlyCharges", "TotalCharges"]   # sayısal sütunlarımız

# GRAFİK 3: kutu grafikleri (kalan ve ayrılan yan yana)
fig, eksenler = plt.subplots(1, 3, figsize=(12, 4))
for ax, sutun in zip(eksenler, sayisal):
    df.boxplot(column=sutun, by="Churn", ax=ax)   # Churn'e göre iki kutu çiz
    ax.set_title(sutun)
    ax.set_xlabel("Churn (0 = kaldı, 1 = ayrıldı)")
fig.suptitle("")                                  # pandas'ın otomatik başlığını sil
plt.tight_layout()
plt.savefig("results/g3_kutu_grafikleri.png", dpi=150)
plt.close()

# Aykırı değer sayısı (IQR kuralı): çok uçta kalan değerleri say
for sutun in sayisal:
    q1, q3 = df[sutun].quantile([0.25, 0.75])     # alt ve üst çeyrek
    iqr = q3 - q1                                 # kutunun genişliği
    alt, ust = q1 - 1.5 * iqr, q3 + 1.5 * iqr     # bu sınırların dışı "aykırı"
    adet = ((df[sutun] < alt) | (df[sutun] > ust)).sum()
    print(sutun, "aykırı değer sayısı:", adet)

# GRAFİK 4: sözleşme türüne göre ayrılma oranı
# Churn 0/1 olduğu için ortalaması = ayrılanların oranı
oran = df.groupby("Contract")["Churn"].mean().sort_values() * 100
oran.plot(kind="bar", color="tab:orange")
plt.ylabel("Ayrılma oranı (%)")
plt.title("Sözleşme türüne göre churn oranı")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/g4_sozlesme_churn.png", dpi=150)
plt.close()

# GRAFİK 5: korelasyon matrisi (sütunlar birbirine ne kadar bağlı?)
korelasyon = df[sayisal + ["SeniorCitizen", "Churn"]].corr()
plt.figure(figsize=(6, 5))
sns.heatmap(korelasyon, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korelasyon matrisi")
plt.tight_layout()
plt.savefig("results/g5_korelasyon.png", dpi=150)
plt.close()

print(korelasyon["Churn"].round(2))               # Churn ile diğerlerinin ilişkisi