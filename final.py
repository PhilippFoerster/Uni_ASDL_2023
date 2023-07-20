import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaModel, RobertaForSequenceClassification, get_linear_schedule_with_warmup
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base", do_lower_case=True)

config = RobertaConfig.from_pretrained("microsoft/codebert-base")
config.num_labels = 1
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)
model.to(device)

max_length = 512

def convert_examples_to_features(js):
    code=' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code, return_tensors="pt", max_length=510, padding="max_length")
    token_count = len(code_tokens)
    code_tokens = code_tokens[:510]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    source_ids = torch.tensor(source_ids)
    return {"input_ids": source_ids, "attention_mask": source_ids.ne(1)}, js['target'], token_count

class CustomDataset(Dataset):
    def __init__(self, file_path, skip):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                input, target, token_count = convert_examples_to_features(js)
                if token_count > 800 and skip:
                    continue
                self.examples.append([input, target])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return self.examples[i][0], self.examples[i][1]

batch_size = 4
train_dataset = CustomDataset('dataset/train_renamed.jsonl', True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
test_dataset = CustomDataset('dataset/test_renamed.jsonl', False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

adam_epsilon = 1e-8
learning_rate = 2e-5
max_grad_norm = 1.0
weight_decay = 0.00
no_decay = ['bias', 'LayerNorm.weight']
epochs = 3
max_steps = epochs * len(train_dataloader)
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)

update_loss_steps = 100

stats = []
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for i, epoch in enumerate(range(epochs)):
    #TRAINING
    bar = tqdm(train_dataloader,total=len(train_dataloader))
    bar.set_description(f"Epoch {epoch}")
    model.train()
    total_loss = 0
    val_results = []
    val_labels = []
    for step, batch in enumerate(bar):
        inputs = batch[0]
        inputs = {key: value.squeeze(1).to(device) for key, value in inputs.items()}
        labels = batch[1].to(device)
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        outputs = model(ids, mask)
        result = outputs.logits[:,0]
        loss = BCEWithLogitsLoss()(result, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        total_loss += loss.mean().item()
        val_results.append(result.cpu().detach().numpy())
        val_labels.append(labels.cpu().numpy())
        if step % update_loss_steps == 99:
            results=np.concatenate(val_results,0)
            labels=np.concatenate(val_labels,0)
            preds=results>0.5
            acc=np.mean(labels==preds)
            loss = total_loss / step
            stats.append({'steps': (epoch*len(train_dataloader))+step, 'loss': loss, 'acc': acc})
            bar.set_description(f"Epoch {epoch}: loss {loss:.4f}, acc {acc:.4f}")
    
    #EVAL
    model.eval()
    with torch.no_grad():
        bar = tqdm(test_dataloader,total=len(test_dataloader))
        total_loss = 0
        val_results = []
        val_labels = []
        for step, batch in enumerate(bar):
            inputs = batch[0]
            inputs = {key: value.squeeze(1).to(device) for key, value in inputs.items()}
            labels = batch[1].to(device)
            ids, mask = inputs['input_ids'], inputs['attention_mask']
            outputs = model(ids, mask)
            result = outputs.logits[:,0]
            loss = BCEWithLogitsLoss()(result, labels.float())
            total_loss += loss.mean().item()
            val_results.append(result.cpu().detach().numpy())
            val_labels.append(labels.cpu().numpy())

    results=np.concatenate(val_results,0)
    labels=np.concatenate(val_labels,0)
    preds=results>0.5
    acc=np.mean(labels==preds)
    loss = total_loss / step
    time.sleep(1) #otherwise screws up tqdm
    print(f"Test set: loss {loss:.4f}, acc {acc:.4f}")
    model.save_pretrained('Models/epoch'+str(epoch))
    test_losses.append(loss)
    test_accs.append(acc)

    #Recompute loss and accuracy on whole training set
    results=np.concatenate(val_results,0)
    labels=np.concatenate(val_labels,0)
    preds=results>0.5
    acc=np.mean(labels==preds)
    loss = total_loss / step
    train_losses.append(stats[-1]['loss'])
    train_accs.append(stats[-1]['acc'])

#Plot loss and accuracy per epoch in 2 subplots next to each other

epochs = list(range(1, len(train_losses) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
fig.suptitle('Loss and accuracy per epoch')
ax1.plot(epochs, train_losses, label='train')
ax1.plot(epochs, test_losses, label='test')
ax1.legend()
ax1.set_title('Loss')
ax2.plot(epochs, train_accs, label='train')
ax2.plot(epochs, test_accs, label='test')
ax2.legend()
ax2.set_title('Accuracy')
plt.savefig('lossacc.png', dpi=500)