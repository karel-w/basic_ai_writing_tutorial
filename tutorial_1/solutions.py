# Complete working example
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import EsmModel, EsmTokenizer
import numpy as np
import pandas as pd

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        inputs = tokenizer(
            sequence, 
            padding='max_length', 
            max_length=512, 
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'labels': torch.tensor(label)
        }

# Model class
class ProteinClassifier(torch.nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(pretrained_model.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get embeddings from pretrained model
        outputs = self.pretrained_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        
        # Average pooling
        pooled_output = torch.mean(sequence_output, dim=1)
        
        # Classification
        return self.classifier(pooled_output)

# Training step
def train_model(model, train_loader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Get batch data (this comes from ProteinDataset)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass (using ProteinClassifier)
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Single predict
def predict_sequence(model, tokenizer, sequence, device='cuda'):
    """
    Make prediction for a single protein sequence
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize sequence
    inputs = tokenizer(
        sequence,
        padding='max_length',
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # Get probabilities
        probs = torch.softmax(outputs, dim=1)
        
        # Get predicted class
        predicted_class = torch.argmax(probs, dim=1)
    
    return {
        'predicted_class': predicted_class.item(),
        'probabilities': probs.squeeze().cpu().numpy()
    }

# Batch predict
def predict_batch(model, tokenizer, sequences, batch_size=32, device='cuda'):
    """
    Make predictions for a list of protein sequences
    """
    # Create dataset
    dataset = ProteinDataset(sequences, labels=[0]*len(sequences))  # dummy labels
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Lists to store predictions
    all_predictions = []
    all_probabilities = []
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probabilities
    }

# Model loading
def load_model(model_path, device='cuda'):
    """
    Load a saved model
    """
    # Load the pretrained model first
    pretrained_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    # Initialize your classifier
    model = ProteinClassifier(pretrained_model, num_labels=2)
    
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    
    # Move to device
    model = model.to(device)
    
    return model

# Initialize tokenizer
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") #We force CPU since we have more memory there

# Example usage with dummy data
sequences = [
    "MLELLPTAVEGVSQAQITGRP",
    "KVFGRCELAAAMKRHGLDNYR"
]
labels = [0, 1]  # Binary classification example

# Create dataset and dataloader
dataset = ProteinDataset(sequences, labels)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
pretrained_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = ProteinClassifier(pretrained_model, num_labels=2)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
train_model(model, train_loader, optimizer, num_epochs=2)
print('saving model')
torch.save(model.state_dict(), 'model_weights.pth')

print('loading model')
model = load_model('model_weights.pth')

# Example sequences
test_sequences = [
    "MLELLPTAVEGVSQAQITGRP",
    "KVFGRCELAAAMKRHGLDNYR",
    "MAEGEITTFTALTEKFNLPPG"
]

# 1. Single sequence prediction
print("\nSingle Sequence Prediction:")
result = predict_sequence(model, tokenizer, test_sequences[0])
print(f"Predicted class: {result['predicted_class']}")
print(f"Class probabilities: {result['probabilities']}")

# 2. Batch prediction
print("\nBatch Prediction:")
results = predict_batch(model, tokenizer, test_sequences)

# Create DataFrame for nice output
df = pd.DataFrame({
    'Sequence': test_sequences,
    'Predicted_Class': results['predictions'],
    'Probability_Class_0': [prob[0] for prob in results['probabilities']],
    'Probability_Class_1': [prob[1] for prob in results['probabilities']]
})

print(df)