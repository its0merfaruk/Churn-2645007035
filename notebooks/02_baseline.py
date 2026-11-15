import sys
sys.path.append(".")

from sklearn.model_selection import train_test_split   # tabloyu ikiye bölen araç
from src.config import SEED
from src.data import load_data

df = load_data()                                       # temizlenmiş tabloyu getir

X = df.drop(columns=["Churn"])                         # X = ipuçları (müşteri özellikleri)
y = df["Churn"]                                        # y = cevap anahtarı (ayrıldı mı?)

# Çalışma kâğıdı (eğitim) %80, sınav kâğıdı (test) %20.
# stratify=y: iki parçada da ayrılan oranı aynı kalsın.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

print("Eğitim:", X_train.shape, "Test:", X_test.shape)
print("Eğitimde ayrılan oranı:", y_train.mean().round(3))
print("Testte ayrılan oranı:", y_test.mean().round(3))

from sklearn.pipeline import Pipeline            # adımları zincirleyen araç
from sklearn.compose import ColumnTransformer    # her sütun grubuna ayrı işlem uygular
from sklearn.impute import SimpleImputer         # boş hücreleri doldurur
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sütunları ikiye ayır: sayılar ve yazılar
sayisal = X.select_dtypes(include="number").columns.tolist()
kategorik = [c for c in X.columns if c not in sayisal]
print("Sayısal sütunlar:", sayisal)
print("Kategorik sütun sayısı:", len(kategorik))

on_isleme = ColumnTransformer([
    # Sayılar: boşu medyanla doldur, sonra aynı ölçeğe getir (z-skor)
    ("sayisal", Pipeline([("doldur", SimpleImputer(strategy="median")),
                          ("olcekle", StandardScaler())]), sayisal),
    # Yazılar: boşu en sık değerle doldur, sonra 0/1 sütunlara çevir
    ("kategorik", Pipeline([("doldur", SimpleImputer(strategy="most_frequent")),
                            ("kodla", OneHotEncoder(handle_unknown="ignore"))]), kategorik),
])

# Sadece EĞİTİM kısmından öğren ve dönüştür
X_train_hazir = on_isleme.fit_transform(X_train)
print("Hazırlanmış eğitim tablosu:", X_train_hazir.shape)

import warnings
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")        # dummy modelin F1 uyarılarını gizle

# İki baseline (temel) model
modeller = {
    "Baseline-0: hep 'kalır' de": DummyClassifier(strategy="prior"),  # hep çoğunluk sınıfını söyler
    "Baseline-1: Karar ağacı (derinlik=3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
}

# 5 katlı çapraz doğrulama: eğitim kısmını 5 parçaya böl, her seferinde 1 parçayla sına
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

satirlar = []
for ad, model in modeller.items():
    # Pipeline: ön işleme + model tek zincirde (her katta ön işleme sadece eğitim parçasından öğrenir)
    boru = Pipeline([("on_isleme", on_isleme), ("model", model)])
    s = cross_validate(boru, X_train, y_train, cv=cv,
                       scoring=["accuracy", "f1", "roc_auc"],
                       return_train_score=True)   # eğitim skoru: aşırı öğrenmeyi görmek için
    satirlar.append({
        "model": ad,
        "accuracy": f"{s['test_accuracy'].mean():.3f} ± {s['test_accuracy'].std():.3f}",
        "F1": f"{s['test_f1'].mean():.3f} ± {s['test_f1'].std():.3f}",
        "ROC-AUC (doğrulama)": f"{s['test_roc_auc'].mean():.3f} ± {s['test_roc_auc'].std():.3f}",
        "ROC-AUC (eğitim)": f"{s['train_roc_auc'].mean():.3f}",
    })

tablo = pd.DataFrame(satirlar)
print(tablo.to_string(index=False))
tablo.to_csv("results/baseline_cv.csv", index=False)   # tabloyu dosyaya kaydet