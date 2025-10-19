import pandas as pd

df = pd.read_csv('StressLevelDataset_capped.csv')
print(df.describe())    

kolom = 'anxiety_level'

# menghitung nilai min dan maks
min_val= df[kolom].min()
max_val= df[kolom].max()

# coba print 
print("hasil dari min dan max value dari kolom anxiety_level adalah")
print(min_val)
print(max_val)

# normalisasi manual menggunakan rumus
df[kolom+ '_minmax']=(df[kolom]-min_val)/(max_val-min_val)
# print buat melihat hasil
print(df[[kolom, kolom + '_minmax']].head())


# menggunakan rumus python
df["anxiety_level"] = (df["anxiety_level"] - df["anxiety_level"].min()) / (df["anxiety_level"].max() - df["anxiety_level"].min())
print(df[["anxiety_level"]].head())


