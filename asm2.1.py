import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. GIẢ LẬP DỮ LIỆU (MOCK DATA)
data = {
    'id': [1, 2, 3, 4, 5, 6],
    'price': [5000, 7500, -100, 150000, 7550, 5100], # Có giá âm và outlier cực lớn
    'area': [50, 70, 45, 500, 72, 51],
    'rooms': [2, 3, 0, 10, 3, 2], # Có số phòng = 0
    'district': ['Quận 1', 'Quận 3', 'Q.1', 'Quận 7', 'Quận 3', 'Quận 1'], # Typo district
    'description': [
        'Nhà đẹp quận 1, gần chợ Bến Thành',
        'Căn hộ cao cấp trung tâm Quận 3, nội thất sang trọng',
        'Nhà giá rẻ',
        'Biệt thự siêu sang view biển',
        'Căn hộ cao cấp tại Q3, đầy đủ nội thất', # Duplicate nội dung với id 2
        'Nhà trung tâm Q1, ngay sát chợ Bến Thành' # Duplicate nội dung với id 1
    ]
}
df = pd.DataFrame(data)

# --- GIAI ĐOẠN 1: XỬ LÝ DỮ LIỆU ---

## 1. Khám phá & Xử lý dữ liệu bẩn
# Chuẩn hóa Typo trong District
df['district'] = df['district'].replace({'Q.1': 'Quận 1', 'Q3': 'Quận 3'})

# Xử lý giá trị bất thường (Giá <= 0 hoặc số phòng = 0)
df = df[(df['price'] > 0) & (df['rooms'] > 0)]

# Điền Missing values (ví dụ minh họa)
df['area'] = df['area'].fillna(df['area'].median())

## 2. Xử lý Outliers (Dùng IQR cho cột Price)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping: Giới hạn giá trị trong khoảng an toàn
df['price_capped'] = np.where(df['price'] > upper_bound, upper_bound, 
                             np.where(df['price'] < lower_bound, lower_bound, df['price']))

## 3. Chuẩn hóa số & Biến đổi Categorical
# Scaling diện tích (Min-Max)
scaler = MinMaxScaler()
df['area_scaled'] = scaler.fit_transform(df[['area']])

# One-hot encoding cho District
df = pd.get_dummies(df, columns=['district'], prefix='dist')

## 4. Phát hiện trùng lặp dựa trên Text Similarity (TF-IDF)
tfidf = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf.fit_transform(df['description'])

# Tính Cosine Similarity giữa các mô tả
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Tìm các cặp có độ tương đồng > 0.6 (ngưỡng giả định)
duplicates = []
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i, j] > 0.6:
            duplicates.append((df.iloc[i]['id'], df.iloc[j]['id'], cosine_sim[i, j]))

print("--- KẾT QUẢ PHÂN TÍCH ---")
print(f"Các cặp bản ghi nghi ngờ trùng lặp: {duplicates}")
print("\nBảng dữ liệu sau khi làm sạch sơ bộ:")
print(df[['id', 'price_capped', 'area_scaled', 'description']].head())# 5. TRỰC QUAN HÓA (Để xem trong VS Code Interactive hoặc save file)
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['price_capped'])
plt.title("Phân phối giá sau khi xử lý Outlier (Capping)")
plt.show()