import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN

df = pd.read_csv("train.csv")

print("Jumlah missing value SEBELUM preprocessing:\n")
print(df.isnull().sum())

# Isi missing value
df_filled = df.copy()
df_filled['Age'] = df_filled['Age'].fillna(df_filled['Age'].median())
df_filled['Embarked'] = df_filled['Embarked'].fillna(df_filled['Embarked'].mode()[0])
df_filled['Cabin'] = df_filled['Cabin'].fillna("Unknown")  # isi Cabin string default

print("\nJumlah missing value SESUDAH preprocessing:\n")
print(df_filled.isnull().sum())

# 2. Transformasi MinMaxScaler
df_num = df_filled.select_dtypes(include=['int64','float64'])
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)

print("\nHasil MinMaxScaler (5 baris pertama):")
print(df_scaled.head())

# 3. Ekstraksi fitur dengan LDA
if "Survived" in df_scaled.columns:
    X = df_scaled.drop(columns=["Survived"])
    y = df_filled["Survived"]

    lda = LDA(n_components=1)  # Untuk target biner hanya bisa 1 komponen
    X_lda = lda.fit_transform(X, y)

    print("\nHasil ekstraksi fitur LDA (5 baris pertama):")
    print(X_lda[:5])

    # ---- Visualisasi hasil ekstraksi LDA ----
    plt.figure(figsize=(8, 4))
    plt.scatter(X_lda, [0]*len(X_lda), c=y, cmap="coolwarm", alpha=0.6)
    plt.title("Visualisasi Ekstraksi Fitur dengan LDA (Sebelum SMOTE-ENN)")
    plt.xlabel("Komponen LDA")
    plt.yticks([])
    plt.colorbar(label="Survived")
    plt.show()

else:
    print("\nKolom target 'Survived' tidak ditemukan di dataset.")

# 4. Tangani Imbalanced Data dengan SMOTE-ENN
if "Survived" in df_filled.columns:
    # Ambil fitur numerik + encode sex & embarked
    df_balancing = df_filled.copy()

    # Encode kolom kategorikal
    df_balancing["Sex"] = df_balancing["Sex"].map({"male": 0, "female": 1})
    df_balancing["Embarked"] = df_balancing["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # membuang kolom non-numerik yang tidak dipakai
    df_balancing = df_balancing.drop(columns=["Name", "Ticket", "Cabin"])

    X = df_balancing.drop(columns=["Survived"])
    y = df_balancing["Survived"]

    print("\nDistribusi kelas sebelum SMOTE-ENN:")
    print(y.value_counts())

    # Plot distribusi sebelum SMOTE
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, palette="viridis")
    plt.title("Distribusi Kelas Sebelum SMOTE-ENN")
    plt.show()

    # menerapkan SMOTE-ENN
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)

    print("\nDistribusi kelas sesudah SMOTE-ENN:")
    print(pd.Series(y_resampled).value_counts())

    # Plot distribusi sesudah SMOTE
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_resampled, palette="viridis")
    plt.title("Distribusi Kelas Setelah SMOTE-ENN")
    plt.show()

    # menyimpan dataset hasil balancing
    df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
    df_balanced["Survived"] = y_resampled
    df_balanced.to_csv("train_balanced.csv", index=False)
    print("\nDataset hasil SMOTE-ENN berhasil disimpan ke train_balanced.csv")
