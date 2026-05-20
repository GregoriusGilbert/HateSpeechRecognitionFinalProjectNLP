import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. Page Configuration
st.set_page_config(page_title="Hate Speech Detector", page_icon="🚫")

@st.cache_resource
def load_model():
    model_name = 'bert-base-uncased' 
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Using 3 labels based on your .pt file mismatch error
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Load weights
    model.load_state_dict(torch.load('model_bert_hatespeech.pt', map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

# Initialize variables
tokenizer, model = None, None

# Load model
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")

# 2. User Interface (English Version)
st.title("🚫 Hate Speech Recognition")
st.write("Enter a sentence to check for hate speech indications.")

user_input = st.text_area("Sentence:", placeholder="Type a sentence to analyze (e.g., I hate this or Have a nice day)")

if st.button("Detect"):
    if not tokenizer or not model:
        st.error("Model or Tokenizer not properly loaded.")
    elif user_input.strip():
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        #2 = Safe, 0/1 = Hate/Abusive
        if prediction == 0:
            st.error(f"Result: **Hate Speech Detected** (Targeting Identity)")
        elif prediction == 1:
            st.warning(f"Result: **Abusive / Offensive Language** (Profanity)")
        else:
            st.success(f"Result: **Safe (Non-Hate Speech)**")
    else:
        st.warning("Please enter some text first.")