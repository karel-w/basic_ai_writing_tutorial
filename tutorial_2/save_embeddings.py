from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from datetime import datetime
import os
import pandas as pd
from transformers import EsmModel, EsmTokenizer

#initialize sequence loading
def load_sequences(data_csv, n_samples):
    df = pd.read_csv(data_csv)
    small_df = df.groupby('label/fitness')['sequence'].apply(lambda s: s.sample(n_samples)).reset_index()

    return small_df['sequence'].tolist(), small_df['label/fitness'].tolist()

train = '../DeepLoc_cls2_data/cls2_normal_train.csv'
valid = '../DeepLoc_cls2_data/cls2_normal_valid.csv'

train_sequences, train_labels = load_sequences(train, 1000)
val_sequences, val_labels = load_sequences(valid, 100)

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

def generate_and_save_embeddings(sequences, labels, model, tokenizer, 
                               batch_size=32, split='train', output_dir='embeddings'):
    """Generate and save embeddings in chunks"""
    
    # Create dataset and dataloader
    dataset = SequenceDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store embeddings and labels
    all_embeddings = []
    all_labels = []
    
    # Process batches
    with torch.no_grad():
        for batch_idx, (batch_sequences, batch_labels) in enumerate(dataloader):
            # Tokenize and generate embeddings
            inputs = tokenizer(batch_sequences, return_tensors='pt', padding=True)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            embeddings = torch.mean(embeddings, dim=1)
            
            # Convert to numpy and store
            batch_embeddings = embeddings.numpy()
            all_embeddings.append(batch_embeddings)
            all_labels.append(batch_labels)
            
            if (batch_idx + 1) % 10 == 0:  # Print progress every 10 batches
                print(f"Processed {(batch_idx + 1) * batch_size} sequences")
    
    # Concatenate all batches
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    
    # Save to disk
    embedding_file = f'{output_dir}/{split}_embeddings_{timestamp}.npy'
    label_file = f'{output_dir}/{split}_labels_{timestamp}.npy'
    
    np.save(embedding_file, final_embeddings)
    np.save(label_file, final_labels)
    
    print(f"Saved embeddings and labels to {output_dir}")
    return embedding_file, label_file

# Usage example:
model_name = "facebook/esm2_t6_8M_UR50D"
pretrained_model = EsmModel.from_pretrained(model_name)
tokenizer = EsmTokenizer.from_pretrained(model_name)

# Generate and save embeddings
embedding_file, label_file = generate_and_save_embeddings(
    sequences=train_sequences,
    labels=train_labels,
    model=pretrained_model,
    tokenizer=tokenizer,
    batch_size=8,  # Adjust based on your GPU memory
    split='train'
)

embedding_file, label_file = generate_and_save_embeddings(
    sequences=val_sequences,
    labels=val_labels,
    model=pretrained_model,
    tokenizer=tokenizer,
    batch_size=8,  # Adjust based on your GPU memory
    split='valid'
)

# Load embeddings (remains the same)
def load_embeddings(embedding_file, label_file):
    """Load embeddings and labels from disk"""
    embeddings = np.load(embedding_file)
    labels = np.load(label_file)
    return embeddings, labels