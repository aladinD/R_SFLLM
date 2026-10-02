import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils.plotting import accumulate_client_metrics, accumulate_master_metrics
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import List, Dict, Literal
import numpy as np
import itertools

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


def fetch_metrics_for_dataset(
        dataset: str, 
        task: str,
        subdirs: list[str],
        base_model: str,
        model_mapping: dict[str, list[str]],
        client_name: str = "client_0",
        plot_data: Literal["client", "master", "both"] = "client"):
        model = model_mapping[base_model][task]
        metric_name = 'acc' if task == 'sc' else 'f1'
        
        scenarios = []
        metrics = []
        xaxes = []
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
                    raise ValueError(f"Error fetching scenario for directory {directory}: {e}")

                # Fetch metrics for the current directory
                _, client_val_df = accumulate_client_metrics(config, client_name + "_logger", config.paths.log_path)
                _, master_val_df = accumulate_master_metrics(config, config.paths.log_path)

                if plot_data in ["client", "both"]:
                    xaxis, metric = client_val_df["epoch"], client_val_df[f"val_{metric_name}"]
                if plot_data in ["master", "both"]:
                    xaxis, metric = master_val_df["epoch"], master_val_df[f"test_{metric_name}"]

                scenarios.append(scenario)
                xaxes.append(xaxis.to_numpy())
                metrics.append(metric.to_numpy())
        return scenarios, xaxes, metrics


def training_curves_barplot(main_dir: str,
                    base_models: str | List[str],
                    sc_datasets: str | List[str],
                    ner_datasets: str | List[str],
                    plot_data: str = "client",
                    client_name: str = "client_0",
                    save_dir: str = ".",
                    plot_name: str = "jointmultiplot.png"
    ):
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
        "None": "No Wireless Channel",
        "no_jammer": "No Attack, only Wireless Impairments",
        "w_protection": "Adversarial Attack with Protection",
        "no_protection": "Adversarial Attack without Protection",
    }

    subdirs: List[str] = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

    if isinstance(sc_datasets, str):
        sc_datasets = [sc_datasets]
    if isinstance(ner_datasets, str):
        ner_datasets = [ner_datasets]
    
    if isinstance(base_models, str):
        base_models = [base_models]
    
    datasets_all = sc_datasets + ner_datasets
    dsets_models = list(itertools.product(datasets_all, base_models))

    metrics_all_client, metrics_all_master = [], []
    for dm in dsets_models:
        dset, mod = dm
        tsk = "sc" if dset in sc_datasets else "ner"
        
        scenarios, xaxes, metrics = fetch_metrics_for_dataset(
            dataset=dset, task=tsk, subdirs=subdirs, 
            base_model=mod, 
            model_mapping=model_mapping, 
            client_name=client_name, 
            plot_data="client"
        )
        d = {s: (x, m) for s, x, m in zip(scenarios, xaxes, metrics)}
        metrics_all_client.append(d)
        print(dm, tsk, scenarios)
        scenarios_master, xaxes_master, metrics_master = fetch_metrics_for_dataset(
            dataset=dset, task=tsk, subdirs=subdirs, 
            base_model=mod, 
            model_mapping=model_mapping, 
            client_name=client_name, 
            plot_data="master"
        )
        ddm = {s: (x, m) for s, x, m in zip(scenarios_master, xaxes_master, metrics_master)}
        metrics_all_master.append(ddm)
    
    # setup everything for clients and master figures
    f_client, ax_client = plt.subplots(len(datasets_all) * len(base_models), 1, figsize=(20, 14))
    ax_client = ax_client.flatten()
    
    f_master, ax_master = plt.subplots(len(datasets_all), len(base_models), figsize=(20, 14))
    ax_master = ax_master.flatten()
    
    clrs = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(scenarios)]
    colors = [clrs[i] for i in [1, 0, 3, 2]] # hacky way to make green and blue first and then yellow and red
    colormap = {k: v for k, v in zip(scenarios, colors)}

    ylabels = []
    for d in dsets_models:
        lb = "Accuracy" if d[0] in sc_datasets else "F1 Score"
        ylabels.append(lb)
    
    for idx, (tmp, a) in enumerate(zip(metrics_all_client, ax_client)):
        # we iterate through the label mappings and get the 
        # required data otherwise the legends will not be synchronized
        for lbl, lbl_map in label_mapping.items():
            xax, mt = tmp[lbl]
            a.plot(xax, mt, linestyle='-', marker='o', color=colormap[lbl], label=lbl_map)
        a.set_xlabel(r'Cumulative Epochs', fontsize=14)
        a.set_ylabel(ylabels[idx], fontsize=14)
        a.set_title(f'{dsets_models[idx][0].upper()} {dsets_models[idx][1].upper()}', fontsize=16)
        # a.set_box_aspect()
        a.grid(True)

    for idx, (tmp, a) in enumerate(zip(metrics_all_master, ax_master)):
        heights = []
        for lbl, lbl_map in label_mapping.items():
            xax, mt = tmp[lbl]
            heights.append(mt[-1])
        a.bar(x=list(range(len(scenarios))), height=heights, label='Global Model', alpha=0.8)
        a.set_title(f'{dsets_models[idx][0].upper()} {dsets_models[idx][1].upper()}', fontsize=16)
    
    f_client.tight_layout()
    f_client.subplots_adjust(bottom=0.1, hspace=0.4)
    f_client.legend(
        list(label_mapping.values()),
        loc='lower center',
        ncol=len(scenarios), 
        fancybox=True, 
        shadow=True,
        fontsize=16
        )

    f_client.savefig(fname=os.path.join(save_dir, f"{plot_name}_plot.pdf"))
    f_master.savefig(fname=os.path.join(save_dir, f"{plot_name}_bar.png"))

    
if __name__ == "__main__":
    training_curves_barplot(main_dir="/home/shared/plotting/", 
                    base_models=['bert', 'roberta'],
                    sc_datasets=['sst2'],
                    ner_datasets=['conll2003'],
                    plot_data='client',
                    client_name="client_0",
                    save_dir = "./",
                    plot_name="figs_icc_single_col")