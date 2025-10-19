import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Langkah 1: Memuat Dataset
df = pd.read_csv('Netflix Dataset.csv')

# Langkah 2: Preprocessing untuk Kolom Numerik
# 2.1: Duration_Minutes (untuk Film)
movies = df[df['Category'] == 'Movie'].copy()
movies['Duration_Minutes'] = movies['Duration'].str.extract(r'(\d+)').astype(float)
# Mengisi missing values dengan median
median_duration = movies['Duration_Minutes'].median()
movies['Duration_Minutes'].fillna(median_duration, inplace=True)
print("\nJumlah Missing Values pada Duration_Minutes setelah Median Imputation:", movies['Duration_Minutes'].isna().sum())

# 2.2: Release_Year (dari Release_Date)
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Release_Year'] = df['Release_Date'].dt.year
# Mengisi missing values dengan modus tahun
year_modus = df['Release_Year'].mode()[0]
df['Release_Year'].fillna(year_modus, inplace=True)
print("\nJumlah Missing Values pada Release_Year setelah Modus Imputation:", df['Release_Year'].isna().sum())

# Langkah 3: Memeriksa Outliers dengan IQR
# Fungsi untuk mendeteksi outliers
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return lower_bound, upper_bound, outliers

# 3.1: Outliers untuk Duration_Minutes
lower_bound_duration, upper_bound_duration, outliers_duration = detect_outliers_iqr(movies, 'Duration_Minutes')
print("\nOutliers untuk Duration_Minutes (Film):")
print("Batas Bawah:", lower_bound_duration)
print("Batas Atas:", upper_bound_duration)
print("Jumlah Outliers:", len(outliers_duration))
print("Contoh Outliers (5 baris pertama):")
print(outliers_duration[['Title', 'Duration_Minutes']].head())

# 3.2: Outliers untuk Release_Year
lower_bound_year, upper_bound_year, outliers_year = detect_outliers_iqr(df, 'Release_Year')
print("\nOutliers untuk Release_Year:")
print("Batas Bawah:", lower_bound_year)
print("Batas Atas:", upper_bound_year)
print("Jumlah Outliers:", len(outliers_year))
print("Contoh Outliers (5 baris pertama):")
print(outliers_year[['Title', 'Release_Year']].head())

# Langkah 4: Visualisasi Outliers
# 4.1: Boxplot dan Histogram untuk Duration_Minutes
plt.figure(figsize=(8, 5))
sns.boxplot(x=movies['Duration_Minutes'])
plt.title('Boxplot Durasi Film (Menit) untuk Mendeteksi Outliers')
plt.xlabel('Durasi (Menit)')
plt.savefig('netflix_duration_boxplot.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(movies['Duration_Minutes'], kde=True, bins=30)
plt.title('Distribusi Durasi Film (Menit)')
plt.xlabel('Durasi (Menit)')
plt.savefig('netflix_duration_distribution.png')
plt.close()

# 4.2: Boxplot dan Histogram untuk Release_Year
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Release_Year'])
plt.title('Boxplot Tahun Rilis untuk Mendeteksi Outliers')
plt.xlabel('Tahun Rilis')
plt.savefig('netflix_year_boxplot.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df['Release_Year'], kde=True, bins=30)
plt.title('Distribusi Tahun Rilis')
plt.xlabel('Tahun Rilis')
plt.savefig('netflix_year_distribution.png')
plt.close()

# Simpan data outliers untuk referensi
outliers_duration.to_csv('netflix_outliers_duration.csv', index=False)
outliers_year.to_csv('netflix_outliers_year.csv', index=False)