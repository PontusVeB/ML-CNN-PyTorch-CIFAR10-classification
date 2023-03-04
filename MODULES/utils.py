import torch
from pathlib import Path
from typing import Dict,List
import matplotlib.pyplot as plt
import pandas as pd

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

def plot_loss_curves_from_dict(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary"""
    #get the loss value of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    
    #get the accuracy values of the ruslts dictionary 
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    
    #epochs
    epochs = range(len(results["train_loss"]))
    
    #setup a plot
    plt.figure(figsize=(15,7))
    
    #plot the loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    
    #plot the accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    
def plot_loss_curves_from_df(df_results: pd.DataFrame):
    """Plots training curves of a results dataframe"""
    
    loss = df_results["train_loss"]
    test_loss = df_results["test_loss"]
    accuracy = df_results["train_acc"]
    test_accuracy = df_results["test_acc"]
    epochs = df_results["epoch"]
    
    #setup a plot
    plt.figure(figsize=(15,7))
    
    #plot the loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    
    #plot the accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    
def make_predictions(model: torch.nn.Module,
                    data: list,
                    device: torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #prepare the sample (ad a batch simenstion and pass to tergget device)
            sample = torch.unsqueeze(sample, dim=0).to(device)
            
            #forward pass ( model outputs raw logits)
            pred_logit= model(sample)
            
            #get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            
            #get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    #stack the pred_probs and turn list into a tensor        
    return torch.stack(pred_probs)
