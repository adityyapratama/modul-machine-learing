import pandas as pd
from sklearn.preprocessing import LabelEncoder

col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv('car_evaluation.csv',header=None, names=col_names)
print(df.head())


# encoding data ke numerik
df_encoded = df.copy()
for col in df_encoded.columns:
    le = LabelEncoder()
    df_encoded[col]=le.fit_transform(df_encoded[col])
    
print(df_encoded.head())

# ekstrasi fitur
x = df_encoded.drop('class',axis=1)
y= df_encoded['class']
