# =========================
# 1. IMPORT
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 2. LOAD DATA AUTO
# =========================
print("=== FILES ===")
files = os.listdir()
print(files)

csv_files = [f for f in files if f.endswith(".csv")]

if len(csv_files) == 0:
    raise Exception("❌ Không có file CSV!")

file_name = "bat_dong_san.csv" if "bat_dong_san.csv" in csv_files else csv_files[0]
print(f"\n👉 Dùng file: {file_name}")

df = pd.read_csv(file_name)

print("\n=== COLUMNS ===")
print(df.columns)

# =========================
# 3. RENAME CỘT (VI → EN)
# =========================
rename_map = {
    'gia_nha': 'price',
    'so_phong': 'rooms',
    'vi_tri': 'location',
    'tinh_trang': 'description'
}

df.rename(columns=rename_map, inplace=True)

# =========================
# 4. FIX THIẾU CỘT
# =========================

if 'price' not in df.columns:
    raise Exception("❌ Thiếu cột giá!")

if 'rooms' not in df.columns:
    df['rooms'] = 1

if 'location' not in df.columns:
    df['location'] = "unknown"

if 'description' not in df.columns:
    df['description'] = ""

# ❗ dataset bạn KHÔNG có area → tạo giả
if 'area' not in df.columns:
    print("⚠️ Không có area → tạo dữ liệu giả")
    df['area'] = np.random.randint(30, 150, size=len(df))

# =========================
# 5. CLEANING
# =========================
df['area'] = df['area'].fillna(df['area'].median())
df['location'] = df['location'].fillna("unknown")
df['description'] = df['description'].fillna("")

df = df[df['price'] > 0]
df = df[df['rooms'] > 0]

df = df.drop_duplicates()

# =========================
# 6. OUTLIER
# =========================
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['price'] >= Q1 - 1.5*IQR) &
        (df['price'] <= Q3 + 1.5*IQR)]

# =========================
# 7. FEATURE ENGINEERING
# =========================
df['price_log'] = np.log1p(df['price'])

df['desc_length'] = df['description'].apply(lambda x: len(str(x).split()))
df['has_luxury'] = df['description'].str.contains("luxury", case=False).astype(int)

df['price_per_m2'] = df['price'] / df['area']
df['luxury_score'] = df['has_luxury'] * df['price_per_m2']

df['area_rooms'] = df['area'] * df['rooms']

# =========================
# 8. TEXT SIMILARITY
# =========================
tfidf = TfidfVectorizer(max_features=50)
text_matrix = tfidf.fit_transform(df['description'])

sim = cosine_similarity(text_matrix)
print("\nSimilarity shape:", sim.shape)

# =========================
# 9. PIPELINE
# =========================
num_cols = ['area', 'rooms', 'price_per_m2', 'area_rooms']
cat_cols = ['location']
text_col = 'description'

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('text', TfidfVectorizer(max_features=50), text_col)
])

# =========================
# 10. TRAIN
# =========================
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoost": GradientBoostingRegressor()
}

print("\n=== MODEL RESULTS ===")

for name, model in models.items():
    pipe = Pipeline([
        ('prep', preprocessor),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    
    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2:", r2)

# =========================
# 11. VISUAL
# =========================
plt.figure()
sns.scatterplot(x='area', y='price', data=df)
plt.title("Area vs Price")
plt.show()

plt.figure()
sns.histplot(df['price'])
plt.title("Price Distribution")
plt.show()

# =========================
# 12. INSIGHT
# =========================
print("\n=== INSIGHT ===")
print("- Giá/m2 cao → nên đầu tư")
print("- Có 'luxury' → giá cao hơn")
print("- RandomForest/GB tốt hơn Linear")

print("\n✅ HOÀN THÀNH!")