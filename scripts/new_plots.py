def generate_single_plot_tikz(main_dir: str, dataset: str, model: str, client_name="client_0", plot_name="singleplot.png"):
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
        # scenario = config.tags[-2] if "tags" in config else None
        # if not scenario:
        #     with open(os.path.join(directory, "tags.log"), 'r') as file:
        #         content = file.read()
        #         tags = [tag.strip() for tag in content.strip("[]").split(",")]
        #         scenario = tags[-1]
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
        # ax = axes[idx // cols, idx % cols] if rows > 1 else axes[idx % cols]

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

        # [existing code to apply client mapping and extract parameters from the configuration]

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
        # [existing code to adjust axis properties]

    # Set labels for the single axis
    ax.set_ylim(0.4, 1)
    
    # Size adjustments
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.set_xlabel('Cumulative Epochs', fontsize=12)

    # ax.set_xlabel('Cumulative Epochs')
    ax.grid(True, alpha=1)
    
    # Add a legend for the whole figure
    # ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=12)

    # Add a single legend for the whole figure
    lines, labels = ax.get_legend_handles_labels()
    # make legend text bigger
    
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