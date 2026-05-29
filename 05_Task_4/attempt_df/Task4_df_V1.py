# %% [markdown]
# # Task 4
# This serves as a template which will guide you through the implementation of this task. It is advised to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps.
# This is the jupyter notebook version of the template. For the python file version, please refer to the file `template_solution.py`.

# %% [markdown]
# First, we import necessary libraries:

# %%
import os
# We disable low-level log outputs by default to keep the terminal clean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Add any other imports you need here

import wandb
from transformers import AutoTokenizer, AutoModel

# %% [markdown]
# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that are not required.

# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64         # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 5          # TODO: Set the number of epochs

# MODEL_NAME = "distilbert-base-uncased"
# MODEL_NAME = "albert-base-v2"
MODEL_NAME = "gpt2"

train_val = pd.read_csv("../train.csv")
test_val = pd.read_csv("../test_no_score.csv")

# %%
# TODO: Fill out SentimentDataset
class SentimentDataset(Dataset):
    def __init__(self, titles, sentences, labels=None):
        self.titles = titles
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, index):
        # TODO: Tokenize and return a {input_ids, attention_mask, labels} dictionary, or just {input_ids, attention_mask} for test data.
        full_text = f"{self.titles[index]} {self.sentences[index]}"
        tokenized_text = self.tokenizer(full_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

        item = {
            "input_ids": tokenized_text["input_ids"].squeeze(0),
            "attention_mask": tokenized_text["attention_mask"].squeeze(0)
        }

        # Handle training data
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)

        return item

# %%
train_dataset = SentimentDataset(
    train_val["title"].tolist(), 
    train_val["sentence"].tolist(), 
    train_val["score"].tolist())
test_dataset = SentimentDataset(
    test_val["title"].tolist(), 
    test_val["sentence"].tolist())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0,         # If you want to utilize multi-processing, set this to the number of your available cores!
                          pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0,          # If you want to utilize multi-processing, set this to the number of your available cores!
                         pin_memory=True)
# Additional code if needed

# %%
# TODO: Fill out SentimentClassifier
class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(MODEL_NAME)

        # Freeze parameters
        for param in self.transformer.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Grab the right token depending on model
        if "gpt2" in MODEL_NAME:
            # GPT-2
            summary_embedding = outputs.last_hidden_state[:, -1, :]
        else:
            # DistilBERT / ALBERT
            summary_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(summary_embedding)
        
        # Return the final score
        return logits.squeeze(-1)


model = SentimentClassifier().to(DEVICE)

# %%
# TODO: Setup loss function, optimizer, and scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = None

# Initialize Wandb
wandb.init(
    project="iml-task4",
    name="run-4-GPT2",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "architecture": MODEL_NAME
    }
)

for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Clear old gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        # Calculate Loss
        loss = criterion(predictions, batch["labels"])
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Log metrics to your shared team dashboard
        wandb.log({
            "train_loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

wandb.finish()

# %%
model.eval()
with torch.no_grad():
    results = []
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Forward pass to get raw scores
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        
        # Convert logits to binary predictions
        predictions = (logits > 0).int()
        
        # Detach from GPU, convert to numpy, and add to list
        results.append(predictions.cpu().numpy())

    with open(f"result_{MODEL_NAME}.txt", "w") as f:
        for val in np.concatenate(results):
            f.write(f"{val}\n")

# %%



