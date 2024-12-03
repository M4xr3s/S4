import torch
from torch.utils.data import DataLoader
from src.model import RegressionModel, ccc_loss
import os

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10, save_path="best_model.pth"):
    best_val_loss = float("inf")  
    for epoch in range(num_epochs):
       
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        val_loss /= len(val_loader.dataset)

       
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)  # Save model state
            print(f"Saved Best Model at Epoch {epoch+1} with Validation Loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
