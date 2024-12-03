from src.feature_extraction import process_files
from src.data_handling import load_folder_data
from src.model import RegressionModel, ccc_loss
from src.train import train_model
from torch.utils.data import DataLoader, TensorDataset
import torch

if __name__ == "__main__":
    

    train_csv_dir = "./data/train/csv"
    train_processed_dir = "./data/train/processed"


    val_csv_dir = "./data/val/csv"
    val_processed_dir = "./data/val/processed"

    process_files(train_audio_dir, train_csv_dir, train_processed_dir)
    process_files(val_audio_dir, val_csv_dir, val_processed_dir)

    train_inputs, train_targets = load_folder_data(train_processed_dir)
    val_inputs, val_targets = load_folder_data(val_processed_dir)
 
    train_dataset = TensorDataset(
        torch.tensor(train_inputs, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_inputs, dtype=torch.float32),
        torch.tensor(val_targets, dtype=torch.float32)
    )

   
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

   
    input_dim = train_inputs.shape[1]
    model = RegressionModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    
    train_model(model, train_loader, val_loader, optimizer, ccc_loss, num_epochs=20)
