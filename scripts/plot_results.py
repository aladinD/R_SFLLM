import os
import pandas as pd

import matplotlib.pyplot as plt
import yaml
from matplotlib.image import imread
from omegaconf import OmegaConf, DictConfig
import tikzplotlib

from typing import List, Dict, Optional

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.plotting import accumulate_client_metrics, accumulate_master_metrics


# Enable LaTeX rendering in Matplotlib
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def fetch_relevant_directories(main_dir: str, dataset: str, model: str) -> list:
    """
    Traverse the main directory and fetch subdirectories that match the dataset and model criteria.
    
    Parameters:
    - main_dir (str): Path to the main directory.
    - dataset (str): Target dataset.
    - model (str): Target model.

    Returns:
    - list: List of subdirectories that match the criteria.
    """
    matched_dirs = []
    
    # List all subdirectories in the main directory
    subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    
    for subdir in subdirs:

        tags_path = os.path.join(main_dir, subdir, "tags.log")

        if os.path.exists(tags_path):
            with open(tags_path, 'r') as file:
                content = file.read()

                # Extracting tags from the content
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                if dataset in tags[0] and model in tags[1]:
                    matched_dirs.append(os.path.join(main_dir, subdir))
    
    return matched_dirs


def fetch_configuration(directory: str) -> DictConfig:
    """
    Fetch and interpret the configuration from `config_tree.yaml` in the given directory.
    
    Parameters:
    - directory (str): Path to the target directory.

    Returns:
    - DictConfig: Configuration dictionary.
    """
    config_path = os.path.join(directory, "config_tree.yaml")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return OmegaConf.create(config)


def save_fig(f, save_path: str | os.PathLike, latex: bool = False, pdf: bool = False, **kwargs):
    """
    Saves the figure to the specified path with the given format options.
    """
    stem = os.path.split(save_path)[0]
    if not os.path.isdir(stem):
        os.mkdir(stem)
    f.savefig(
        os.path.join(save_path + ".png"),
        dpi=150,
        bbox_inches="tight",
        **kwargs,
    )
    if pdf:
        f.savefig(
            os.path.join(save_path + ".pdf"),
            dpi=150,
            bbox_inches="tight",
            **kwargs,
        )
    if latex:
        tikzplotlib.save(
            os.path.join(save_path + ".tex"),
            figure=f,
            strict=False,
            **kwargs,
        )


def generate_multiplot(main_dir: str, dataset: str, model: str, client_name="client_0", plot_name="multiplot.png"):
    """
    Generate a multiplot for all the subfolders that match the given dataset and model.
    """
    # Define known scenarios
    known_scenarios = ["None", "no_jammer", "no_protection", "w_protection"]

    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)
    
    # Number of directories fetched
    num_dirs = len(directories)
    
    # Calculate the layout for the subplots
    cols = 2
    rows = (num_dirs + 1) // cols
    
    # Create the main figure for multiplot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    for idx, directory in enumerate(directories):
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Fetch the scenario from the config or tags.log
        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in known_scenarios:
                    scenario = tag
                    break

        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                for tag in tags:
                    if tag in known_scenarios:
                        scenario = tag
                        break

        # Determine the current subplot axis
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]

        # Plot the metrics for the current directory on the current subplot axis
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        # Extracting parameters from the configuration
        task = config.task_name
        if task == "sc":
            # Plot val metrics for client and master model for sequence classification
            ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{client_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label='Global SFL Model', color='red', linestyle='-', marker='o')
            
            ax.set_ylabel('Accuracy')
        
        elif task == "ner":
            # Plot F1 score metrics for client and master model for NER            
            ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{client_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label='Global SFL Model', color='red', linestyle='-', marker='o')
            
            ax.set_ylabel('F1 Score')
        
        # Set the y-axis scale
        ax.set_ylim(0, 0.7)

        # Set scenario titles
        title_mapping = {
            "None": "Baseline Performance without the Wireless Channel",
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Jamming without Protection",
            "w_protection": "Adversarial Jamming with Protection",
        }
        ax.set_title(title_mapping.get(scenario, "Baseline Performance without the Wireless Channel"))
        
        ax.set_xlabel('Cumulative Epochs')
        ax.grid(True, alpha=0.5)
    
    # Remove any unused subplots
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    # Add a single legend for the whole figure
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=2)

    # Title
    main_title_fontsize = 20
    subtext_fontsize = 12
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

    if task == "sc":
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        main_title = 'F1 Scores across Global Rounds and Epochs'

    # fig.suptitle('Classification Accuracies across Global Rounds and Epochs', fontsize=20)
    subtext = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}" 
    
    fig.suptitle(main_title, fontsize=main_title_fontsize)
    fig.text(0.5, 0.92, subtext, ha='center', va='center', fontsize=subtext_fontsize)

    # Tight layout and save the multiplot
    plt.tight_layout()

    # Adjust spacing and layout
    plt.subplots_adjust(bottom=0.1, top=0.85)

    plt.savefig(os.path.join(main_dir, plot_name))


def generate_multiplot_tikz(main_dir: str, dataset: str, model: str, client_name="client_0", plot_name="multiplot.png"):
    """
    Generate a multiplot for all the subfolders that match the given dataset and model.
    """
    # Define known scenarios
    known_scenarios = ["None", "no_jammer", "no_protection", "w_protection"]

    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)
    
    # Number of directories fetched
    num_dirs = len(directories)
    
    # Calculate the layout for the subplots
    cols = 2
    rows = (num_dirs + 1) // cols
    
    # Create the main figure for multiplot
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    for idx, directory in enumerate(directories):
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Fetch the scenario from the config or tags.log
        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in known_scenarios:
                    scenario = tag
                    break

        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                for tag in tags:
                    if tag in known_scenarios:
                        scenario = tag
                        break

        # Determine the current subplot axis
        ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]

        # Plot the metrics for the current directory on the current subplot axis
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        # CLient mapping
        client_mapping = {
            "client_0": "Client 1",
            "client_1": "Client 2",
            "client_2": "Client 3",
        }

        # Apply client mapping to client_name
        display_name = client_mapping.get(client_name, client_name)

        # Extracting parameters from the configuration
        task = config.task_name
        if task == "sc":
            # Plot val metrics for client and master model for sequence classification
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_acc'].to_numpy(), label=f'{display_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_acc'].to_numpy(), label='Global SFL Model', color='red', linestyle='-', marker='o')


            # ax.set_ylabel('Accuracy')
            ax.set_ylabel('Accuracy', fontsize=12)
        
        elif task == "ner":
            # Plot F1 score metrics for client and master model for NER              
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_f1'].to_numpy(), label=f'{display_name}', color='blue', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_f1'].to_numpy(), label='Global SFL Model', color='red', linestyle='-', marker='o')


            ax.set_ylabel('F1 Score', fontsize=12)
        
        # Set the y-axis scale
        ax.set_ylim(0, 1)

        # Set scenario titles
        title_mapping = {
            "None": "SFL Baseline",
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Worst-Case Jamming without Protection",
            "w_protection": "Adversarial Worst-Case Jamming with Protection",
        }
        ax.set_title(title_mapping.get(scenario, "Baseline Performance without the Wireless Channel"))
        
        # Size adjustments
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        ax.set_xlabel('Cumulative Epochs', fontsize=12)

        # ax.set_xlabel('Cumulative Epochs')
        ax.grid(True, alpha=1)
    
    # Remove any unused subplots
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes.flatten()[j])

    # Add a single legend for the whole figure
    lines, labels = ax.get_legend_handles_labels()
    
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=2, fontsize=12)

    # Title
    main_title_fontsize = 20
    subtext_fontsize = 12
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]

    # make adataset name all capital
    dataset_name = dataset_name.upper()

    # make model name all capital
    model = model.upper()

    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds
    batch_size = config.datamodule.batch_size

    if task == "sc":
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        main_title = 'F1 Scores across Global Rounds and Epochs'

    plt.tight_layout()

    # Adjust spacing and layout
    plt.subplots_adjust(bottom=0.1, top=0.85)

    # plt.savefig(os.path.join(main_dir, plot_name))
    tikz_save_path = os.path.join(main_dir, "multiplot.tex")
    tikzplotlib.save(tikz_save_path)
    print(tikz_save_path)

    save_fig(fig, save_path=main_dir, latex=True, pdf=True)


def generate_joint_plot(main_dir: str, 
                        dataset: str, 
                        model: str, 
                        plot_data: str = "client", 
                        client_name="client_0", 
                        plot_name="jointplot.png",
                        ax=None):
    """
    Generate a joint plot for all subfolders that match the given dataset and model.
    
    Parameters:
    - main_dir: Main directory path.
    - dataset: Dataset name.
    - model: Model name.
    - plot_data: A string which can be "client", "master", or "both" to decide which data to plot.
    - client_name: The name of the client.
    - plot_name: The name of the saved plot.
    
    """
    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)

    # Create the main figure for the joint plot
    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax_provided = False

    # Loop through each directory and plot the data
    for directory in directories:
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Handle potential errors while fetching scenario
        try:
            scenario = config.tags[-1] if "tags" in config else None
            if not scenario:
                with open(os.path.join(directory, "tags.log"), 'r') as file:
                    content = file.read()
                    tags = [tag.strip() for tag in content.strip("[]").split(",")]
                    scenario = tags[-1]
        except Exception as e:
            print(f"Error fetching scenario for directory {directory}: {e}")
            continue
        
        # Fetch metrics for the current directory
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)
        
        # Extracting task from the configuration
        task = config.get("task_name", "")
        
        # Plot client data based on the task
        label_mapping = {
            "None": "Baseline Performance without the Wireless Channel",
            "no_jammer": "No Jamming, only Wireless Impairments",
            "no_protection": "Adversarial Jamming without Protection",
            "w_protection": "Adversarial Jamming with Protection",
        }
        label_scenario = label_mapping.get(scenario, scenario)

        if plot_data in ["client", "both"]:
            if task == "sc":
                ax.plot(client_val_df['epoch'], client_val_df['val_acc'], label=f'{label_scenario}', linestyle='-', marker='o')
            elif task == "ner":
                ax.plot(client_val_df['epoch'], client_val_df['val_f1'], label=f'{label_scenario}', linestyle='-', marker='o')
        
        # Plot master data based on the task
        if plot_data in ["master", "both"]:
            if task == "sc":
                ax.plot(master_val_df['epoch'], master_val_df['test_acc'], label=f'{label_scenario}', linestyle='--', marker='x')
            elif task == "ner":
                ax.plot(master_val_df['epoch'], master_val_df['test_f1'], label=f'{label_scenario}', linestyle='--', marker='x')

    # Set the axis parameters
    ax.set_ylim(0.4, 1)
    ax.set_xlabel('Cumulative Epochs')

    # Set title
    main_title_fontsize = 16
    subtitle_fontsize = 10
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

    if task == "sc":
        ax.set_ylabel('Accuracy')
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        ax.set_ylabel('F1 Score')
        main_title = 'F1 Scores across Global Rounds and Epochs'

    subtitle = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}"
    plt.suptitle(main_title, fontsize=main_title_fontsize)  
    plt.title(subtitle, fontsize=subtitle_fontsize) 

    ax.grid(True, alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=4)
    plt.subplots_adjust(bottom=0.25)


    # Adjust spacing and layout
    plt.tight_layout()

    # plt.savefig(os.path.join(main_dir, plot_name))
    if not ax_provided:
        plt.savefig(os.path.join(main_dir, plot_name))


def generate_joint_multiplot(main_dir: str,
                             base_model: str,
                             sc_datasets: List[str],
                             ner_datasets: List[str],
                             plot_data: str = "client",
                             client_name: str = "client_0",
                             save_dir: str = "./",
                             plot_name: str = "jointmultiplot.png"):
    """
    Generate a joint multiplot for all subfolders that match the given datasets and model.
    
    Parameters:
    - main_dir: Main directory path.
    - base_model: Base model name (e.g., 'roberta' or 'bert').
    - sc_datasets: List of SC dataset names.
    - ner_datasets: List of NER dataset names.
    - plot_data: A string which can be "client", "master", or "both" to decide which data to plot.
    - client_name: The name of the client.
    - plot_name: The name of the saved plot.
    """
    
    # Model Mapping
    model_mapping: Dict[str, Dict[str, str]] = {
        'roberta': {
            'sc': 'roberta_for_sequence_classification',
            'ner': 'roberta_for_token_classification'
        },
        'bert': {
            'sc': 'bert_for_sequence_classification',
            'ner': 'bert_for_token_classification'
        }
    }
    
    # Label Mapping
    label_mapping: Dict[str, str] = {
        "None": "Baseline Performance without the Wireless Channel",
        "no_jammer": "No Jamming, only Wireless Impairments",
        "no_protection": "Adversarial Jamming without Protection",
        "w_protection": "Adversarial Jamming with Protection"
    }

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(20, 14))
    
    # Get the list of subdirectories in main_dir
    subdirs: List[str] = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    # Initialize an empty set to store unique legend labels
    legend_entries = {}

    # Function to handle plotting for both SC and NER datasets
    def plot_data_for_dataset(row: int, col: int, dataset: str, task: str):
        model = model_mapping[base_model][task]
        metric = 'acc' if task == 'sc' else 'f1'
        ylabel = r'Accuracy' if task == 'sc' else r'F1 Score'
        
        for subdir in subdirs:
            directories = fetch_relevant_directories(subdir, dataset, model)
            if not directories:
                continue

            for directory in directories:
                # Fetch the configuration for the current directory
                config = fetch_configuration(directory)

                # Handle relative paths
                relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
                relative_path = os.path.join(*relative_path_components[3:4])
                relative_path = os.path.join(relative_path, 'logs/')  
                config.paths.log_path = os.path.join(subdir, relative_path)

                # Get scenario
                try:
                    scenario = config.tags[-1] if "tags" in config else None
                    if not scenario:
                        with open(os.path.join(directory, "tags.log"), 'r') as file:
                            content = file.read()
                            tags = [tag.strip() for tag in content.strip("[]").split(",")]
                            scenario = tags[-1]
                except Exception as e:
                    print(f"Error fetching scenario for directory {directory}: {e}")
                    continue

                # Fetch metrics for the current directory
                client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
                master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

                # Plot data
                ax = axs[row][col]
                label_scenario = label_mapping.get(scenario, scenario)

                label = f'{label_scenario}'
                if plot_data in ["client", "both"]:
                    line1, = ax.plot(client_val_df['epoch'], client_val_df[f'val_{metric}'], label=f'{label_scenario}', linestyle='-', marker='o')
                    legend_entries[label] = line1
                
                if plot_data in ["master", "both"]:
                    line2, = ax.plot(master_val_df['epoch'], master_val_df[f'test_{metric}'], label=f'{label_scenario}', linestyle='--', marker='x')
                    legend_entries[label] = line2

                # Set the axis parameters for the subplot
                ax.set_ylim(0., 1) if task == 'sc' or dataset == "conll2003" else ax.set_ylim(0.05, 0.5)
                ax.set_xlabel(r'Cumulative Epochs', fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)
                ax.set_title(f'{dataset.upper()}',  fontsize=16)
                ax.grid(True, alpha=0.5)

    # Process SC datasets
    for idx, dataset in enumerate(sc_datasets):
        plot_data_for_dataset(0, idx, dataset, 'sc')

    # Process NER datasets
    for idx, dataset in enumerate(ner_datasets):
        plot_data_for_dataset(1, idx, dataset, 'ner')

    fig.legend(handles=legend_entries.values(), 
               labels=legend_entries.keys(), 
               loc='lower center', 
               ncol=len(legend_entries), 
               fancybox=True, 
               shadow=True,
               fontsize=14)
    
    fig.suptitle(r'Performance Metrics for SC and NER NLP Tasks', fontsize=18, fontweight='bold')


    # Set global title, layout, and save figure
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.09)  
    plt.savefig(os.path.join(save_dir, plot_name))


def generate_single_plot_tikz(main_dir: str, 
                              dataset: str, 
                              model: str, 
                              client_name="client_0", 
                              plot_name="singleplot.png"):
    """
    Generate a single plot for all subfolders that match the given dataset and model.
    """
    # Define known scenarios
    known_scenarios = ["None", "no_jammer", "no_protection", "w_protection"]

    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)

    # Create the main figure for the single plot
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, directory in enumerate(directories):
        # Fetch the configuration for the current directory
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        # Fetch the scenario from the config or tags.log
        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in known_scenarios:
                    scenario = tag
                    break

        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                for tag in tags:
                    if tag in known_scenarios:
                        scenario = tag
                        break

        # Plot the metrics for the current directory on the single subplot axis
        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        # CLient mapping
        client_mapping = {
            "client_0": "Client 1",
            "client_1": "Client 2",
            "client_2": "Client 3",
        }

        # Apply client mapping to client_name
        display_name = client_mapping.get(client_name, client_name)

        # Add plots to the single axis
        task = config.task_name
        if task == "sc":
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_acc'].to_numpy(), label=f'{display_name} ({scenario})', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_acc'].to_numpy(), label=f'Global SFL Model ({scenario})', linestyle='-', marker='o')
            ax.set_ylabel('Accuracy', fontsize=12)
        
        elif task == "ner":
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_f1'].to_numpy(), label=f'{display_name} ({scenario})', linestyle='-', marker='o')
            ax.plot(master_val_df['epoch'].to_numpy(), master_val_df['test_f1'].to_numpy(), label=f'Global SFL Model ({scenario})', linestyle='-', marker='o')

            ax.set_ylabel('F1 Score', fontsize=12)

    # Set labels for the single axis
    ax.set_ylim(0.4, 1)
    
    # Size adjustments
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('Cumulative Epochs', fontsize=12)

    # ax.set_xlabel('Cumulative Epochs')
    ax.grid(True, alpha=1)
    
    # Add a single legend for the whole figure
    lines, labels = ax.get_legend_handles_labels()
    
    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=2, fontsize=12)

    # Title
    main_title_fontsize = 20
    subtext_fontsize = 12
    model = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if task == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]

    # make adataset name all capital
    dataset_name = dataset_name.upper()

    # make model name all capital
    model = model.upper()

    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds
    batch_size = config.datamodule.batch_size

    if task == "sc":
        main_title = 'Classification Accuracies across Global Rounds and Epochs'

    elif task == "ner":
        main_title = 'F1 Scores across Global Rounds and Epochs'

    # fig.suptitle('Classification Accuracies across Global Rounds and Epochs', fontsize=20)
    subtext = f"Model Type: {model}, Model Task: {model_task}, Dataset: {dataset_name}, Batch Size: {batch_size}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}" 
    
    fig.suptitle(main_title, fontsize=main_title_fontsize)
    fig.text(0.5, 0.92, subtext, ha='center', va='center', fontsize=subtext_fontsize)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, top=0.85)

    # Save the single plot
    tikz_save_path = os.path.join(main_dir, "singleplot.tex")
    tikzplotlib.save(tikz_save_path)
    print(tikz_save_path)
    save_fig(fig, save_path=os.path.join(main_dir, plot_name), latex=True, pdf=True)


def arrange_plots_into_large_figure(plot_paths: list, output_path: str):
    """
    Arrange given PNG plot paths into a large 5x2 layout and save as a single image.
    
    Args:
    - plot_paths (list): List of paths to the PNG plot files.
    - output_path (str): Path where the resulting large plot should be saved.
    """
    # Create a 5x2 subplot layout
    fig, axes = plt.subplots(5, 2, figsize=(20, 40))  # Adjust the figure size as needed
    plt.subplots_adjust(hspace=0.0, wspace=0.0)  # Adjust spacing as needed
    
    for idx, plot_path in enumerate(plot_paths):
        ax = axes[idx // 2, idx % 2]
        img = imread(plot_path)
        ax.imshow(img)
        ax.axis('off')  # Hide the axis
    
    # Save the large figure to the specified output path
    plt.savefig(output_path + "/plot.png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    print(f"Large plot arranged and saved to {output_path}")


def generate_jointplot_tikz(main_dir: str, 
                            dataset: str, 
                            model: str, 
                            client_name="client_0", 
                            plot_name="jointplot.png", 
                            lims: list = [0.4, 1]):
    """
    Generate a joint plot for all subfolders that match the given dataset and model, using LaTeX font.
    """
    # Font size configuration
    common_font_size = 16
    axis_ticks_font_size = 16

    # Define known scenarios and their display names
    scenario_display_names = {
        "None": "SFL Baseline",
        "no_jammer": "Gaussian",
        "no_protection": "No Protection",
        "w_protection": "Protection"
    }

    # Fetch all relevant directories based on the dataset and model
    directories = fetch_relevant_directories(main_dir, dataset, model)

    # Create the main figure for joint plot
    fig, ax = plt.subplots(figsize=(15, 10))

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    client_mapping = {
        "client_0": "Client 1",
        "client_1": "Client 2",
        "client_2": "Client 3",
    }

    # Assume configuration is the same for all directories
    if directories:
        config = fetch_configuration(directories[0])

    model_name = config.tags[1].split("_")[0]
    model_task = "Sequence Classification" if config.task_name == "sc" else "Named Entity Recognition"
    dataset_name = config.tags[0]
    num_labels = config.model.config.num_labels
    num_clients = config.sfl.num_clients
    num_epochs = config.sfl.num_epochs
    num_rounds = config.sfl.num_rounds

    for idx, directory in enumerate(directories):
        config = fetch_configuration(directory)

        # Handle relative paths
        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in scenario_display_names:
                    scenario = tag
                    break

        if not scenario:
            with open(os.path.join(directory, "tags.log"), 'r') as file:
                content = file.read()
                tags = [tag.strip() for tag in content.strip("[]").split(",")]
                for tag in tags:
                    if tag in scenario_display_names:
                        scenario = tag
                        break

        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        display_name = client_mapping.get(client_name, client_name)

        color = colors[idx % len(colors)]

        if config.task_name == "sc":
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_acc'].to_numpy(), label=f'{display_name} - {scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
            ax.set_ylabel('Accuracy', fontsize=common_font_size)
        elif config.task_name == "ner":
            ax.plot(client_val_df['epoch'].to_numpy(), client_val_df['val_f1'].to_numpy(), label=f'{display_name} - {scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
            ax.set_ylabel('F1 Score', fontsize=common_font_size)

        ax.set_ylim(lims)


    ax.tick_params(axis='both', labelsize=axis_ticks_font_size)

    main_title = 'Classification Accuracies across Global Rounds and Epochs' if config.task_name == "sc" else 'F1 Scores across Global Rounds and Epochs'
    subtitle = f"Model Type: {model_name}, Model Task: {model_task}, Dataset: {dataset_name}, Number of Labels: {num_labels}, Number of Clients: {num_clients}, Number of Epochs: {num_epochs}, Number of SFL Rounds: {num_rounds}"

    main_title_fontsize = 32
    subtitle_fontsize = 14
    # plt.suptitle(main_title,fontsize=main_title_fontsize)
    # plt.title(subtitle, fontsize=subtitle_fontsize)
    plt.title(main_title, fontsize=main_title_fontsize, pad=20)

    ax.set_xlabel('Cumulative Epochs', fontsize=common_font_size)
    ax.grid(True, alpha=1)

    # Position the legend outside the plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4, fontsize=common_font_size)

    # Tight layout and save the joint plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.85)

    tikz_save_path = os.path.join(main_dir, "jointplot.tex")
    tikzplotlib.save(tikz_save_path)
    print(tikz_save_path)

    save_fig(fig, os.path.join(main_dir, plot_name), latex=True, pdf=True)


def plot_on_axes(ax, main_dir, dataset, model, client_name, lims, title, caption, common_font_size=16):
    scenario_display_names = {
        "None": "SFL Baseline",
        "no_jammer": "Gaussian",
        "no_protection": "No Protection",
        "w_protection": "Protection"
    }

    directories = fetch_relevant_directories(main_dir, dataset, model)
    # colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown']
    client_mapping = {
        "client_0": "Client 1",
        "client_1": "Client 2",
        "client_2": "Client 3",
    }

    if directories:
        config = fetch_configuration(directories[0])

    for idx, directory in enumerate(directories):
        config = fetch_configuration(directory)

        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in scenario_display_names:
                    scenario = tag
                    break

        client_train_df, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)

        display_name = client_mapping.get(client_name, client_name)
        color = colors[idx % len(colors)]

        task_label = 'Accuracy' if config.task_name == "sc" else 'F1 Score'
        metric_data = client_val_df['val_acc'] if config.task_name == "sc" else client_val_df['val_f1']

        # ax.plot(client_val_df['epoch'].to_numpy(), metric_data.to_numpy(), label=f'{display_name} - {scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
        ax.plot(client_val_df['epoch'].to_numpy(), metric_data.to_numpy(), label=f'{scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
        ax.set_ylabel(task_label, fontsize=common_font_size)
        # ax.set_title(title, fontsize=common_font_size, pad=20)

    ax.set_ylim(lims)
    ax.tick_params(axis='both', labelsize=common_font_size)
    ax.grid(True, alpha=1)
    ax.set_xlabel('Cumulative Epochs', fontsize=common_font_size)
    # ax.set_xlabel(caption, fontsize=common_font_size - 2)  # Slightly smaller font for caption
    ax.text(0.5, -0.25, caption, fontsize=common_font_size - 2, fontweight='bold', ha='center', va='top', transform=ax.transAxes)


def plot_global_on_axes(ax, main_dir, dataset, model, lims, title, caption, common_font_size=16):
    scenario_display_names = {
        "None": "SFL Baseline",
        "no_jammer": "Gaussian",
        "no_protection": "No Protection",
        "w_protection": "Protection"
    }

    directories = fetch_relevant_directories(main_dir, dataset, model)
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown']

    if directories:
        config = fetch_configuration(directories[0])

    for idx, directory in enumerate(directories):
        config = fetch_configuration(directory)

        relative_path_components = config.paths.client_log_path.split('/logs/', 1)[-1].split("/")
        relative_path = os.path.join(*relative_path_components[3:4])
        relative_path = os.path.join(relative_path, 'logs/')  
        config.paths.log_path = os.path.join(main_dir, relative_path)

        master_train_df, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

        scenario = None
        if "tags" in config:
            for tag in config.tags:
                if tag in scenario_display_names:
                    scenario = tag
                    break

        color = colors[idx % len(colors)]

        task_label = 'Accuracy' if config.task_name == "sc" else 'F1 Score'
        metric_data = master_val_df['test_acc'] if config.task_name == "sc" else master_val_df['test_f1']

        # ax.plot(master_val_df['epoch'].to_numpy(), metric_data.to_numpy(), label=f'Global SFL Model - {scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
        ax.plot(master_val_df['epoch'].to_numpy(), metric_data.to_numpy(), label=f'{scenario_display_names[scenario]}', color=color, linestyle='-', marker='o')
        ax.set_ylabel(task_label, fontsize=common_font_size)

    ax.set_ylim(lims)
    # ax.set_title(title, fontsize=common_font_size, pad=20)
    ax.tick_params(axis='both', labelsize=common_font_size)
    ax.grid(True, alpha=1)
    ax.set_xlabel('Cumulative Epochs', fontsize=common_font_size)
    ax.text(0.5, -0.25, caption, fontsize=common_font_size - 2, ha='center', va='top', transform=ax.transAxes, fontweight='bold')


def generate_3x2_figure(main_dirs, datasets, models, client_names, lims, titles, captions, common_font_size=16):
    fig, axes = plt.subplots(3, 2, figsize=(30, 20))

    for i, ax in enumerate(axes.flatten()):
        if i < len(main_dirs):
            plot_on_axes(ax, main_dirs[i], datasets[i], models[i], client_names[i], lims[i], titles[i], captions[i], common_font_size)
        else:
            ax.axis('off')

    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.15, bottom=0.1, top=0.9)  # Adjust the spacing

    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=common_font_size)

    tikz_save_path = "combined_3x2_plot.tex"
    tikzplotlib.save(tikz_save_path)

    print(tikz_save_path)

    save_fig(fig, os.path.join("/home/aladin/latest/resilient_sfl", tikz_save_path), latex=True, pdf=True)


def generate_5x2_figure(main_dirs, datasets, models, client_names, lims, titles, captions, common_font_size=16, master: bool=False):
    fig, axes = plt.subplots(5, 2, figsize=(30, 30))  # Changed to 5x2 grid

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            index = i * 2 + j  # Calculate the index for the current subplot
            if index < len(main_dirs):
                if master:
                    plot_global_on_axes(ax, main_dirs[index], datasets[index], models[index], lims[index], titles[index], captions[index], common_font_size)
                else:
                    plot_on_axes(ax, main_dirs[index], datasets[index], models[index], client_names[index], lims[index], titles[index], captions[index], common_font_size)
            else:
                ax.axis('off')

    plt.subplots_adjust(hspace=0.45, wspace=0.15, bottom=0.09, top=0.9)  # Adjust the spacing

    # Since this figure will be larger, ensure the legend is appropriately placed
    lines, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=4, fontsize=common_font_size)

    tikz_save_path = "combined_5x2_plot.tex"
    tikzplotlib.save(tikz_save_path)

    print(tikz_save_path)

    save_fig(fig, os.path.join("/home/aladin/latest/resilient_sfl", tikz_save_path), latex=True, pdf=True)


if __name__ == "__main__":

    generate_multiplot_tikz(
        main_dir='/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_bert_per_batch_adversarial_bs64', 
        dataset='sst2', 
        model='bert_for_sequence_classification',
        client_name="client_0",
        plot_name="sst2_bert_per_batch_adversarial_NEW.png"
        )

    # generate_jointplot_tikz(
    #     main_dir='/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_bert_per_batch_adversarial_bs64', 
    #     dataset='sst2', 
    #     model='bert_for_sequence_classification',
    #     lims = [0.4, 1],
    #     client_name="client_0",
    #     plot_name="sst2_bert_per_batch_adversarial.png"
    #     )
    
    # generate_single_plot_tikz(
    # main_dir='/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_bert_per_round_adversarial_bs64', 
    # dataset='sst2', 
    # model='bert_for_sequence_classification',
    # client_name="client_0",
    # plot_name="singleplot_sst2_bert.png")




    # 3x2 plot
    main_dirs = ['/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_bert_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/mnli_latest/mnli_roberta_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/qnli_latest/qnli_bert_per_batch_adversarial_bs32',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/qnli_latest/qnli_roberta_per_batch_adversarial_bs32',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/wnut_latest/wnut_bert_per_batch_adversarial_bs4',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/wnut_latest/wnut_roberta_per_batch_adversarial_bs4'
    ]

    datasets = ['sst2', 'mnli', 'qnli', 'qnli', 'wnut_17', 'wnut_17']
    models = ['bert_for_sequence_classification', 
              'roberta_for_sequence_classification', 
              'bert_for_sequence_classification', 
              'roberta_for_sequence_classification', 
              'bert_for_token_classification', 
              'roberta_for_token_classification'] 
    
    client_names = ['client_0', 
                    'client_0', 
                    'client_0', 
                    'client_0', 
                    'client_0', 
                    'client_2']
    
    lims = [[0.4, 1],
            [0.2, 1],
            [0.4, 1],
            [0.4, 1],
            [0.1, 0.7],
            [0.1, 0.7]
            ]
    
    titles = ['Test1',
                'Test2',
                'Test3',
                'Test4',
                'Test5',
                'Test6']
    
    captions = ['a) Fine-Tuning BERT on SST2 (Client 1 Results)',
                'b) Fine-Tuning RoBERTa on MNLI (Client 1 Results)',
                'c) Fine-Tuning BERT on QNLI (Client 1 Results)',
                'd) Fine-Tuning RoBERTa on QNLI (Client 1 Results)',
                'e) Fine-Tuning BERT on WNUT17 (Client 1 Results)',
                'f) Fine-Tuning RoBERTa on WNUT17 (Client 3 Results)']

    generate_3x2_figure(main_dirs, 
                        datasets, 
                        models, 
                        client_names, 
                        lims, 
                        titles,
                        captions,
                        common_font_size=25)
    

    # 5x2 plot
    main_dirs = ['/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_bert_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/sst2_latest/sst2_roberta_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/mnli_latest/mnli_bert_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/mnli_latest/mnli_roberta_per_batch_adversarial_bs64',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/qnli_latest/qnli_bert_per_batch_adversarial_bs32',
                 '/home/aladin/latest/resilient_sfl/logs/sc/multiruns/qnli_latest/qnli_roberta_per_batch_adversarial_bs32',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/conll_latest/conll_bert_per_batch_adversarial_bs16',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/conll_latest/conll_roberta_per_batch_adversarial_bs16',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/wnut_latest/wnut_bert_per_batch_adversarial_bs4',
                 '/home/aladin/latest/resilient_sfl/logs/ner/multiruns/wnut_latest/wnut_roberta_per_batch_adversarial_bs4'
    ]

    datasets = ['sst2', 'sst2', 'mnli', 'mnli', 'qnli', 'qnli', 'conll', 'conll', 'wnut_17', 'wnut_17']
    models = ['bert_for_sequence_classification', 
              'roberta_for_sequence_classification', 
              'bert_for_sequence_classification', 
              'roberta_for_sequence_classification',
              'bert_for_sequence_classification', 
              'roberta_for_sequence_classification',
              'bert_for_token_classification', 
              'roberta_for_token_classification',
              'bert_for_token_classification', 
              'roberta_for_token_classification'] 
    
    client_names = ['client_0', 
                    'client_0', 
                    'client_0', 
                    'client_0', 
                    'client_0', 
                    'client_0',
                    'client_0', 
                    'client_0',
                    'client_0', 
                    'client_1']
    
    lims = [[0.4, 1],
            [0.4, 1],

            [0.2, 1],
            [0.2, 1],

            [0.4, 1],
            [0.4, 1],

            [0, 1],
            [0, 1],

            [0, 0.7],
            [0, 0.7]
            ]
    
    
    titles = ['Test1',
                'Test2',
                'Test3',
                'Test4',
                'Test5',
                'Test6',
                'Test7',
                'Test8',
                'Test9',
                'Test10']
    
    captions = ['a) Results for Fine-Tuning BERT on SST2',
                'b) Results for Fine-Tuning RoBERTa on SST2',
                'c) Results for Fine-Tuning BERT on MNLI',
                'd) Results for Fine-Tuning RoBERTa on MNLI',
                'e) Results for Fine-Tuning BERT on QNLI',
                'f) Results for Fine-Tuning RoBERTa on QNLI',
                'g) Results for Fine-Tuning BERT on CONLL2003',
                'h) Results for Fine-Tuning RoBERTa on CONLL2003',
                'i) Results for Fine-Tuning BERT on WNUT17',
                'j) Results for Fine-Tuning RoBERTa on WNUT17']
    

    generate_5x2_figure(main_dirs, 
                        datasets, 
                        models, 
                        client_names, 
                        lims, 
                        titles,
                        captions,
                        common_font_size=25,
                        master=True)