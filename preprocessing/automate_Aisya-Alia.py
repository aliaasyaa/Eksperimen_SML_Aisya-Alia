# =============================================================
# automate_Aisya-Alia.py
# Skrip otomasi dari Eksperimen_MSML_Aisya-Alia.ipynb
# Dataset : Healthcare Stroke (healthcare_stroke_raw.csv)
# =============================================================

# =====================================================
# SECTION 2 — IMPORT LIBRARY
# =====================================================

# Library dasar
import pandas as pd
import numpy as np

# Visualisasi
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (cocok untuk skrip)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Utilitas
import warnings
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 100

# Direktori output untuk gambar & hasil preprocessing
OUTPUT_DIR   = 'output_visualisasi'
PREPROC_DIR  = 'preprocessing'
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(PREPROC_DIR, exist_ok=True)

print("=" * 50)
print("LIBRARY BERHASIL DIIMPORT")
print("=" * 50)


# =====================================================
# SECTION 3 — MEMUAT DATASET
# =====================================================

print("\n" + "=" * 50)
print("MEMUAT DATASET")
print("=" * 50)

df = pd.read_csv('healthcare_stroke_raw.csv', sep=';')

print(f"Dataset berhasil dimuat!")
print(f"Dimensi dataset : {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"Kolom           : {df.columns.tolist()}")

print("\n--- 5 Baris Pertama ---")
print(df.head())

print("\n--- Info Dataset ---")
print(df.info())

print("\n--- Statistik Deskriptif ---")
print(df.describe(include='all'))


# =====================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS (EDA)
# =====================================================

print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 50)

# --------------------------------------------------
# 4.1 Info & Shape
# --------------------------------------------------
print("\n" + "=" * 50)
print("INFO DATASET")
print("=" * 50)
print(df.info())
print("\nShape dataset:", df.shape)

# --------------------------------------------------
# 4.2 Statistik Deskriptif
# --------------------------------------------------
print("\n" + "=" * 50)
print("STATISTIK DESKRIPTIF")
print("=" * 50)
print(df.describe())

# --------------------------------------------------
# 4.3 Cek Missing Values
# --------------------------------------------------
print("\n" + "=" * 50)
print("CEK MISSING VALUES")
print("=" * 50)

missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("Tidak ada missing values.")
else:
    print("Ada missing values.")

# --------------------------------------------------
# 4.4 Distribusi Fitur Numerik (Histogram)
# --------------------------------------------------
print("\n" + "=" * 50)
print("DISTRIBUSI FITUR NUMERIK")
print("=" * 50)

num_cols = ['age', 'avg_glucose_level', 'bmi']

df[num_cols].hist(bins=20, figsize=(12, 6), edgecolor='black')
plt.suptitle("Histogram Fitur Numerik")
plt.tight_layout()
hist_path = os.path.join(OUTPUT_DIR, '01_histogram_numerik.png')
plt.savefig(hist_path)
plt.close()
print(f"Histogram disimpan → {hist_path}")

# --------------------------------------------------
# 4.5 Distribusi Target
# --------------------------------------------------
print("\n" + "=" * 50)
print("DISTRIBUSI TARGET (stroke)")
print("=" * 50)

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='stroke')
plt.title("Distribusi Kelas Stroke")
plt.xlabel("Stroke")
plt.ylabel("Jumlah Data")
target_path = os.path.join(OUTPUT_DIR, '02_distribusi_target.png')
plt.savefig(target_path)
plt.close()
print(f"Plot target disimpan → {target_path}")
print(df['stroke'].value_counts())

# --------------------------------------------------
# 4.6 Distribusi Fitur Kategorik
# --------------------------------------------------
print("\n" + "=" * 50)
print("DISTRIBUSI FITUR KATEGORIK")
print("=" * 50)

cat_cols = [
    'gender', 'ever_married', 'work_type',
    'Residence_type', 'smoking_status',
    'hypertension', 'heart_disease'
]

plt.figure(figsize=(18, 12))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df, x=col)
    plt.title(col)
    plt.xticks(rotation=30)
plt.tight_layout()
cat_path = os.path.join(OUTPUT_DIR, '03_distribusi_kategorik.png')
plt.savefig(cat_path)
plt.close()
print(f"Plot kategorik disimpan → {cat_path}")

# --------------------------------------------------
# 4.7 Heatmap Korelasi
# --------------------------------------------------
print("\n" + "=" * 50)
print("HEATMAP KORELASI")
print("=" * 50)

numeric_df = df.select_dtypes(include=[np.number])
corr        = numeric_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
corr_path = os.path.join(OUTPUT_DIR, '04_heatmap_korelasi.png')
plt.savefig(corr_path)
plt.close()
print(f"Heatmap disimpan → {corr_path}")

# --------------------------------------------------
# 4.8 Deteksi Outlier (Boxplot)
# --------------------------------------------------
print("\n" + "=" * 50)
print("DETEKSI OUTLIER DENGAN BOXPLOT")
print("=" * 50)

plt.figure(figsize=(12, 4))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
box_path = os.path.join(OUTPUT_DIR, '05_boxplot_outlier.png')
plt.savefig(box_path)
plt.close()
print(f"Boxplot disimpan → {box_path}")


# =====================================================
# SECTION 5 — DATA PREPROCESSING
# =====================================================

print("\n" + "=" * 50)
print("PREPROCESSING")
print("=" * 50)

# --------------------------------------------------
# 5.1 Copy Dataset
# --------------------------------------------------
df_clean = df.copy()
print("Shape awal:", df_clean.shape)

# --------------------------------------------------
# 5.2 Hapus Kolom Tidak Relevan (id)
# --------------------------------------------------
df_clean.drop(columns=['id'], inplace=True)
print("Kolom setelah drop id:")
print(df_clean.columns.tolist())

# --------------------------------------------------
# 5.3 Penanganan Missing Values
# --------------------------------------------------
print("\n=== PENANGANAN MISSING VALUES ===")

missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()

if missing_cols:
    for col in missing_cols:
        if df_clean[col].dtype in ['float64', 'int64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"{col} diisi median = {median_val}")
        else:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"{col} diisi modus = {mode_val}")
else:
    print("Tidak ada missing values.")

# --------------------------------------------------
# 5.4 Hapus Data Duplikat
# --------------------------------------------------
print("\n=== HAPUS DUPLIKAT ===")

duplicate_count = df_clean.duplicated().sum()
print("Jumlah duplikat:", duplicate_count)

df_clean.drop_duplicates(inplace=True)
print("Shape setelah hapus duplikat:", df_clean.shape)

# --------------------------------------------------
# 5.5 Encoding Data Kategorik
# --------------------------------------------------
print("\n=== ENCODING DATA KATEGORIK ===")

encoders = {}

for col in cat_cols[:5]:   # gender, ever_married, work_type, Residence_type, smoking_status
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    encoders[col] = le
    print(f"{col} berhasil di-encode")

# --------------------------------------------------
# 5.6 Binning Age
# --------------------------------------------------
print("\n=== BINNING AGE ===")

df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 18, 35, 50, 65, 100],
    labels=['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
)

age_encoder = LabelEncoder()
df_clean['age_group'] = age_encoder.fit_transform(df_clean['age_group'])
print("Binning age berhasil")

# --------------------------------------------------
# 5.7 Deteksi & Penanganan Outlier (IQR Capping)
# --------------------------------------------------
print("\n=== PENANGANAN OUTLIER ===")

outlier_cols = ['age', 'avg_glucose_level', 'bmi']

for col in outlier_cols:
    Q1    = df_clean[col].quantile(0.25)
    Q3    = df_clean[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    df_clean[col] = np.where(df_clean[col] < lower, lower, df_clean[col])
    df_clean[col] = np.where(df_clean[col] > upper, upper, df_clean[col])

    print(f"{col}: lower={lower:.2f}, upper={upper:.2f}")

# --------------------------------------------------
# 5.8 Pisahkan Fitur & Target
# --------------------------------------------------
X = df_clean.drop(columns=['stroke'])
y = df_clean['stroke']

# --------------------------------------------------
# 5.9 Standarisasi Fitur
# --------------------------------------------------
print("\n=== STANDARISASI ===")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print("Shape setelah scaling:", X_scaled_df.shape)

# --------------------------------------------------
# 5.10 Gabungkan & Simpan Hasil Preprocessing
# --------------------------------------------------
df_preprocessed = pd.concat(
    [X_scaled_df, y.reset_index(drop=True)],
    axis=1
)

output_csv = os.path.join(PREPROC_DIR, 'stroke_preprocessed.csv')
df_preprocessed.to_csv(output_csv, index=False)

print(f"\nDataset berhasil disimpan → {output_csv}")

print("\nContoh hasil preprocessing:")
print(df_preprocessed.head())

# =====================================================
# SELESAI
# =====================================================
print("\n" + "=" * 50)
print("PIPELINE SELESAI")
print("=" * 50)
print(f"Visualisasi  : {OUTPUT_DIR}/")
print(f"Preprocessed : {output_csv}")
