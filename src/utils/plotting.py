import pandas as pd
from omegaconf import DictConfig
import os
import matplotlib.pyplot as plt


def accumulate_client_metrics(cfg: DictConfig, client_name: str, logs_path: str) -> None:
    """
    Reads and accumulates metrics for a specified client from all rounds.
    """
    # Hyperparameters
    task = cfg.task_name
    num_epochs = cfg.sfl.num_epochs
    num_rounds = cfg.sfl.num_rounds

    client_dir = os.path.join(logs_path, 'clients', client_name)

    # List all rounds for the client
    rounds = [d for d in os.listdir(client_dir) if os.path.isdir(os.path.join(client_dir, d))]
    rounds = rounds[:num_rounds]
    rounds.sort(key=lambda x: int(x.split("_")[-1])) # sort rounds ascendingly

    all_train_metrics = []
    all_val_metrics = []
    for idx, r in enumerate(rounds):
        metrics_path = os.path.join(client_dir, r, 'metrics.csv')
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            
            # Update the epoch number by adding an offset
            df['epoch'] = df['epoch'] + idx * num_epochs
            
            # Split the metrics into training and validation
            if task == "sc":
                train_metrics = df[['epoch', 'train_loss', 'train_acc']]
                val_metrics = df[['epoch', 'val_loss', 'val_acc']]
            elif task == "ner":
                train_metrics = df[['epoch', 'train_loss', 'train_f1', 'train_precision', 'train_recall']]
                val_metrics = df[['epoch', 'val_loss', 'val_f1', 'val_precision', 'val_recall']]
            else:
                raise ValueError("Task should be one of ner, sc!")
            
            all_train_metrics.append(train_metrics)
            all_val_metrics.append(val_metrics)
    
    # Concatenate metrics from all rounds
    train_df = pd.concat(all_train_metrics, ignore_index=True)
    val_df = pd.concat(all_val_metrics, ignore_index=True)
    
    # Drop rows where values are NaN
    if task == "sc":	
        train_df.dropna(subset=['train_acc'], inplace=True)        
        val_df.dropna(subset=['val_acc'], inplace=True)
    elif task == "ner":
        train_df.dropna(subset=['train_f1'], inplace=True)
        val_df.dropna(subset=['val_f1'], inplace=True)
    else:
        print("ERROR")

    # Reset index
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    return train_df, val_df


# def accumulate_master_metrics(cfg: DictConfig, logs_path: str) -> None:
#     """
#     Reads and accumulates metrics for the master model from all rounds.
    
#     Parameters:
#         base_path (str): Base directory where master logs are stored.
        
#     Returns:
#         DataFrame: Accumulated metrics.
#     """
#     # Hyperparameters
#     num_epochs = cfg.sfl.num_epochs
#     num_rounds = cfg.sfl.num_rounds

#     base_path = logs_path + "master/"
#     train_dir = os.path.join(base_path, 'train')
#     val_dir = os.path.join(base_path, 'validation')
    
#     # List all rounds for the master model
#     rounds = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
#     rounds = rounds[:num_rounds]
#     rounds.sort(key=lambda x: int(x.split("_")[-1])) # sort rounds ascendingly
    
#     all_train_metrics = []
#     all_val_metrics = []
#     for idx, r in enumerate(rounds):
#         train_metrics_path = os.path.join(train_dir, r, 'metrics.csv')
#         val_metrics_path = os.path.join(val_dir, r, 'metrics.csv')
        
#         if os.path.exists(train_metrics_path):
#             train_df = pd.read_csv(train_metrics_path)
#             # Update the epoch number based on the round number
#             train_df['epoch'] = train_df['epoch'] + idx * num_epochs -1
#             all_train_metrics.append(train_df)
        
#         if os.path.exists(val_metrics_path):
#             val_df = pd.read_csv(val_metrics_path)
#             # Update the epoch number based on the round number
#             val_df['epoch'] = val_df['epoch'] + idx * num_epochs -1
#             all_val_metrics.append(val_df)
    
#     # Concatenate metrics from all rounds
#     accumulated_train_df = pd.concat(all_train_metrics, ignore_index=True)
#     accumulated_val_df = pd.concat(all_val_metrics, ignore_index=True)

#     # Adjust master epoch numbering
#     accumulated_train_df['epoch'] = accumulated_train_df['epoch'] + num_epochs
#     accumulated_val_df['epoch'] = accumulated_val_df['epoch'] + num_epochs
    
#     return accumulated_train_df, accumulated_val_df


def accumulate_master_metrics(cfg: DictConfig, logs_path: str):
    """
    Reads and accumulates metrics for the master model from all rounds.
    
    Parameters:
        cfg (DictConfig): Configuration object.
        logs_path (str): Base directory where master logs are stored.
        
    Returns:
        Tuple[Optional[pd.DataFrame], pd.DataFrame]: Accumulated training and validation metrics.
    """
    # Hyperparameters
    num_epochs = cfg.sfl.num_epochs
    num_rounds = cfg.sfl.num_rounds

    base_path = logs_path + "master/"
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'validation')
    
    # List all rounds for the master model
    rounds = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    rounds = rounds[:num_rounds]
    rounds.sort(key=lambda x: int(x.split("_")[-1])) # sort rounds ascendingly
    
    all_train_metrics = []
    all_val_metrics = []
    for idx, r in enumerate(rounds):
        train_metrics_path = os.path.join(train_dir, r, 'metrics.csv')
        val_metrics_path = os.path.join(val_dir, r, 'metrics.csv')
        
        if os.path.exists(train_metrics_path):
            train_df = pd.read_csv(train_metrics_path)
            # Update the epoch number based on the round number
            train_df['epoch'] = train_df['epoch'] + idx * num_epochs -1
            all_train_metrics.append(train_df)
        
        if os.path.exists(val_metrics_path):
            val_df = pd.read_csv(val_metrics_path)
            # Update the epoch number based on the round number
            val_df['epoch'] = val_df['epoch'] + idx * num_epochs -1
            all_val_metrics.append(val_df)
    
    # Concatenate metrics from all rounds
    accumulated_train_df = pd.concat(all_train_metrics, ignore_index=True) if all_train_metrics else None
    accumulated_val_df = pd.concat(all_val_metrics, ignore_index=True) if all_val_metrics else None

    # Adjust master epoch numbering for training and validation
    if accumulated_train_df is not None:
        accumulated_train_df['epoch'] = accumulated_train_df['epoch'] + num_epochs
    
    if accumulated_val_df is not None:
        accumulated_val_df['epoch'] = accumulated_val_df['epoch'] + num_epochs

    return accumulated_train_df, accumulated_val_df


def plot_metrics(cfg: DictConfig,
                 plot_name: str ='result.png',
                 client_name: str = 'client_0_logger',
                 save_dir: str = None,
                 logs_path: str = None, 
                 plot_train_metrics: bool = False) -> None:
    """
    Plots the training and validation metrics for a specified client and the master model.
    """
    # Hyperparameters
    task = cfg.task_name
    main_title_fontsize = 16
    subtitle_fontsize = 10

    # Dataframes
    client_train_df, client_val_df = accumulate_client_metrics(cfg, client_name, logs_path)
    master_train_df, master_val_df = accumulate_master_metrics(cfg, logs_path)
    
    # Plotting
    plt.figure(figsize=(10, 6))

    if task == "sc":
        # Plot val metrics for client and master model for sequence classification
        plt.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_acc'].to_numpy(), label=f'{client_name.replace("_logger", "")} per epoch val accuracies', color='blue', linestyle='-', marker='o')
        plt.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_acc'].to_numpy(), label='global val accuracy after each round', color='red', linestyle='-', marker='o')

        # Plot acc metrics for client and master model
        if plot_train_metrics:
            plt.plot(client_train_df['epoch'].to_numpy(), client_train_df['train_acc'].to_numpy(), label=f'{client_name.replace("_logger", "")} per epoch train accuracies', linestyle='-', marker='o')
            plt.plot(master_train_df['epoch'].to_numpy(), master_train_df['test_acc'].to_numpy(), label='global train accuracy after each round', linestyle='-', marker='o')
    
        # Labels and titles
        plt.ylabel('Accuracy')
        main_title = 'Accuracies across Global Rounds and Epochs'
        subtitle = f"Model Type: {cfg.model.config.pretrained_model_name_or_path}, Dataset: {cfg.datamodule.glue_dataset}, Number of Clients: {cfg.sfl.num_clients}"
        
        plt.suptitle(main_title, fontsize=main_title_fontsize)  
        plt.title(subtitle, fontsize=subtitle_fontsize) 

    elif task == "ner":
        # Plot F1 score metrics for client and master model for NER
        plt.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_f1'].to_numpy(), label=f'{client_name.replace("_logger", "")} per epoch val F1', color='blue', linestyle='-', marker='o')
        plt.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_f1'].to_numpy(), label='global val F1 after each round', color='red', linestyle='-', marker='o')

        # Plot F1 score for training metrics
        if plot_train_metrics:
            plt.plot(client_train_df['epoch'].to_numpy(), client_train_df['train_f1'].to_numpy(), label=f'{client_name.replace("_logger", "")} per epoch train F1', linestyle='-', marker='o')
            plt.plot(master_train_df['epoch'].to_numpy(), master_train_df['test_f1'].to_numpy(), label='global train F1 after each round', linestyle='-', marker='o')

        # Labels and titles
        plt.ylabel('F1 Score')
        main_title = 'F1 Scores across Global Rounds and Epochs'
        subtitle = f"Model Type: {cfg.model.config.pretrained_model_name_or_path}, Dataset: {cfg.datamodule.ner_dataset}, Number of Clients: {cfg.sfl.num_clients}"
        
        plt.suptitle(main_title, fontsize=main_title_fontsize)  
        plt.title(subtitle, fontsize=subtitle_fontsize) 
    
    plt.xlabel('Cumulative Epochs')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.savefig(os.path.join(save_dir, plot_name))