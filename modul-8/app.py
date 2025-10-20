# import pandas as pd
# import re

# # 1. Baca file CSV
# df = pd.read_csv("emails.csv")

# # 2. Tentukan kolom teks (ganti jika nama kolom berbeda)
# text_col = "text"  # sesuaikan dengan nama kolom isi email di CSV kamu

# # 3. Gabungkan semua teks menjadi satu string panjang
# all_text = " ".join(df[text_col].astype(str))

# # 4. Tokenisasi (pecah per kata, hanya huruf)
# tokens = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())

# # 5. Simpan hasil tokenisasi ke DataFrame baru
# df_tokens = pd.DataFrame(tokens, columns=["word"])

# # 6. Simpan ke file baru
# df_tokens.to_csv("emails_tokenized.csv", index=False)

# print("✅ File berhasil ditokenisasi. Hasil disimpan di 'emails_tokenized.csv'")
# print(df_tokens.head(20))

import pandas as pd
import re
from collections import Counter


df = pd.read_csv("emails.csv")

# coloumn dataset
text_col = "text"
label_col = "spam"

# memisahkan coloumn yang spam dan bukan spam
spam_emails = df[df[label_col] == 1]
nonspam_emails = df[df[label_col] == 0]

# Tokenisasi
def tokenize(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

spam_words = []
for text in spam_emails[text_col]:
    spam_words.extend(tokenize(str(text)))

nonspam_words = []
for text in nonspam_emails[text_col]:
    nonspam_words.extend(tokenize(str(text)))

# Hitung frekuensi 
spam_counts = Counter(spam_words)
nonspam_counts = Counter(nonspam_words)

# Semua kata unik
all_words = set(spam_counts.keys()).union(set(nonspam_counts.keys()))

# Hitung probabilitas Bayes 
total_spam_words = sum(spam_counts.values())
total_nonspam_words = sum(nonspam_counts.values())
total_words = total_spam_words + total_nonspam_words

p_spam = len(spam_emails) / len(df)
p_nonspam = len(nonspam_emails) / len(df)

data = []

for word in all_words:
    p_word_given_spam = spam_counts[word] / total_spam_words if total_spam_words > 0 else 0
    p_word_given_nonspam = nonspam_counts[word] / total_nonspam_words if total_nonspam_words > 0 else 0
    p_word = (spam_counts[word] + nonspam_counts[word]) / total_words if total_words > 0 else 0

    # Bayes probability
    if p_word > 0:
        p_spam_given_word = (p_word_given_spam * p_spam) / p_word
        p_nonspam_given_word = (p_word_given_nonspam * p_nonspam) / p_word
    else:
        p_spam_given_word = 0

    data.append({
        "word": word,
        "count_in_spam": spam_counts[word],
        "count_in_nonspam": nonspam_counts[word],
        "P(word)" : round(p_word,6),
        "P(non-spam)" : round(p_nonspam,6),
        "P(spam)" : round(p_spam,6),
        "P(word|spam)" : round(p_word_given_spam,6),
        "P(word|nonspam)" : round(p_word_given_nonspam,6),
        "P(spam|word)": round(p_spam_given_word, 6),
        "P(non_spam|word)" : round(p_nonspam_given_word,6)
    })

# menyimpan hasil ke CSV
df_bayes = pd.DataFrame(data)
df_bayes.sort_values(by="P(spam|word)", ascending=False, inplace=True)
df_bayes.to_csv("emails_bayes.csv", index=False)

print("✅ Hasil disimpan di 'emails_bayes.csv'")
print(df_bayes.head(20))
