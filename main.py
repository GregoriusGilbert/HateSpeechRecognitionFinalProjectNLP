import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Dataset
df = pd.read_csv('data/labeled_data.csv')
print("✅ Dataset successfully loaded!")

# 2. Preprocessing Function
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'^rt\s+', '', text)
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

df['clean_text'] = df['tweet'].apply(clean_tweet)

# 3. Data Splitting (80:10:10)
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['class']
)

val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    random_state=42, 
    stratify=temp_df['class']
)

print("\n--- Dataset Splitting Results ---")
print(f"Training Set   : {len(train_df)} rows")
print(f"Validation Set : {len(val_df)} rows")
print(f"Testing Set    : {len(test_df)} rows")

# 4. BERT Tokenizer Setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sample_text = train_df['clean_text'].iloc[0]
tokens = tokenizer.tokenize(sample_text)

print("\n--- BERT Tokenization Preview ---")
print(f"Original Text: {sample_text}")
print(f"Tokens: {tokens}")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Feature Extraction (TF-IDF)
print("\n--- Training Baseline Model (Logistic Regression) ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['class']
y_test = test_df['class']

# 2. Train Logistic Regression
baseline_model = LogisticRegression(max_iter=1000, class_weight='balanced')
baseline_model.fit(X_train, y_train)

# 3. Evaluation
y_pred = baseline_model.predict(X_test)

print("\n--- Baseline Model Evaluation Results ---")
print(classification_report(y_test, y_pred))

# --- Model 2: SVM (Support Vector Machine) ---
print("\n--- Training Model 2: SVM ---")
# Kita pakai LinearSVC karena cepat dan bagus untuk data teks
svm_model = LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM Evaluation Results:")
print(classification_report(y_test, y_pred_svm))

# --- Model 3: Random Forest ---
print("\n--- Training Model 3: Random Forest ---")
# n_estimators=100 artinya kita pakai 100 pohon keputusan
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Evaluation Results:")
print(classification_report(y_test, y_pred_rf))