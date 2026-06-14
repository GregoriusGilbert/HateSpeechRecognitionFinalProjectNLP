import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import joblib 

# 1. Load Dataset
df = pd.read_csv('data/merged_randomized_with_unnamed_sorted.csv')
print("✅ Dataset successfully loaded!")

# 2. Preprocessing Function
def clean_tweet(text):
    # case folding
    text = text.lower()
    # rt removal
    text = re.sub(r'^rt\s+', '', text)
    # @ removal
    text = re.sub(r'@[\w]*', '', text)
    # url removal
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # non alphanumeric filter
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text

df['clean_text'] = df['tweet'].astype(str).apply(clean_tweet)

# 3. Data Splitting (80:20)
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['class']
)

print("\n--- Dataset Splitting Results ---")
print(f"Training Set : {len(train_df)} rows")
print(f"Testing Set  : {len(test_df)} rows")

# 4. BERT Tokenizer Setup 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sample_text = train_df['clean_text'].iloc[0]
tokens = tokenizer.tokenize(sample_text)

print("\n--- BERT Tokenization Preview ---")
print(f"Original Text: {sample_text}")
print(f"Tokens: {tokens}")

# 5. Feature Extraction (TF-IDF)
print("\n--- Training Baseline Models ---")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['class']
y_test = test_df['class']

class_labels = ['Hate Speech', 'Offensive', 'Safe']

# --- MODEL 1: LOGISTIC REGRESSION ---
print("\n[Executing Model 1: Logistic Regression...]")
baseline_model = LogisticRegression(max_iter=1000, class_weight='balanced')
baseline_model.fit(X_train, y_train)
y_pred_logreg = baseline_model.predict(X_test)

print("\n--- Baseline (LogReg) Evaluation Results ---")
print(classification_report(y_test, y_pred_logreg))

# Confusion Matrix LR
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Logistic Regression (Baseline)', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Actual Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_logreg.png', dpi=300)
plt.close() # Menghapus plot dari memori setelah disimpan
print("📊 Generated: 'confusion_matrix_logreg.png'")


# --- MODEL 2: SVM ---
print("\n[Executing Model 2: SVM...]")
svm_model = LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

print("\n--- SVM Evaluation Results ---")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Support Vector Machine (SVM)', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Actual Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_svm.png', dpi=300)
plt.close()
print("📊 Generated: 'confusion_matrix_svm.png'")


# --- MODEL 3: RANDOM FOREST ---
print("\n[Executing Model 3: Random Forest...]")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Evaluation Results ---")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Purples', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix - Random Forest Classifier', fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Actual Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_rf.png', dpi=300)
plt.close()
print("📊 Generated: 'confusion_matrix_rf.png'")


print("\n✅ All baseline models trained, evaluated, and matrices plotted!")

# --- 6. SAVE MODEL & VECTORIZER UNTUK STREAMLIT ---
joblib.dump(baseline_model, 'logistic_model.pkl', compress=3)
joblib.dump(svm_model, 'svm_model.pkl', compress=3)
joblib.dump(rf_model, 'rf_model.pkl', compress=3) 
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl', compress=3)
print("📦 All models and vectorizer successfully saved as compressed .pkl files!")