from main import train_df, val_df, tokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW  
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

# 1. Create Dataset Class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # FIX: Indentasi harus masuk ke dalam def __getitem__
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 2. Prepare Data Loaders
BATCH_SIZE = 16 
MAX_LEN = 64

train_data = HateSpeechDataset(
    texts=train_df['clean_text'].to_numpy(),
    labels=train_df['class'].to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# 3. Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=3 
)
model.to(device)

print("\n--- BERT Model Ready for Fine-Tuning ---")

# 4. Optimizer & Scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3 
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# 5. Training Function
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    
    for d in tqdm(data_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return sum(losses) / len(losses)

print("\n--- Starting Training (This will take time on CPU) ---")
avg_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
print(f"Epoch 1 Complete. Average Loss: {avg_loss}")