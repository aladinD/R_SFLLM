from omegaconf import OmegaConf
import itertools
import copy
import os
import rootutils

def main(experiment_dir: str = os.path.split(__file__)[0], 
         adversarial: bool = True,
         noise_mode: str = "per_round", 
         mse_base_path: str = "/home/aladin/latest/resilient_sfl/mse_files" ) -> None:
    """
    Generates experiment configs for all tasks
    """
    # Define available task configurations
    all_dict = {
        # "ner": {
        #     "datamodule": ["conll2003.yaml", "conll2012_ontonotesv5.yaml", "wnut_17.yaml"],
        #     "model": ["bert_for_token_classification.yaml", "roberta_for_token_classification.yaml"],
        #     "wireless": [None, "no_jammer.yaml", "no_protection.yaml", "w_protection.yaml"]
        # }
        "sc": {
            "datamodule": ["sst2.yaml", "cola.yaml", "mnli.yaml", "mrpc.yaml", "qnli.yaml", "rte.yaml"], 
            "model": ["bert_for_sequence_classification.yaml", "roberta_for_sequence_classification.yaml"],
            "wireless": [None, "no_jammer.yaml", "no_protection.yaml", "w_protection.yaml"]
        }
    }

    # Base configuration scaffold
    config_scaffold = {
        "defaults": [{f"override /{o}": None} for o in all_dict["sc"].keys()],
        "tags": [],
        "task_name": None,
        "mse_path": None,
        "noise_mode": None,
        "adversarial": None
    }
    print(config_scaffold)

    mse_path_map = {
            "no_jammer.yaml": mse_base_path + "/mse_no_jammer.npy",
            "no_protection.yaml": mse_base_path + "/mse_no_protection.npy",
            "w_protection.yaml": mse_base_path + "/mse_w_protection.npy",
            None: None
        }

    # Loop through tasks and their configurations and generate experiment configs
    for task, vals in all_dict.items():

        task_dir = experiment_dir
        # # Create task directory if it doesn't exist
        # task_dir = os.path.join(experiment_dir, task)
        # if not os.path.isdir(task_dir):
        #     os.mkdir(task_dir)
        
        # Extract individual configurations
        dmodules = vals["datamodule"]
        models = vals["model"]
        wireless = vals["wireless"]
        names = list(vals.keys())

        # Generate combinations of configurations
        combinations = itertools.product(dmodules, models, wireless)
        for comb in combinations:
            conf_dict = copy.deepcopy(config_scaffold)  # Copy the base scaffold
            conf_dict["task_name"] = task
            tags = list(map(lambda x: x.split(".")[0] if isinstance(x, str) else str(x), comb))

            # Add adversarial and noise mode to tags
            if adversarial:
                tags.append('adversarial')

            if noise_mode:
                tags.append(noise_mode)

            conf_dict["tags"] = tags
            for i, c in enumerate(comb):
                conf_dict["defaults"][i][f"override /{names[i]}"] = c

            # Set mse_path based on the wireless value
            conf_dict["mse_path"] = mse_path_map[comb[names.index("wireless")]]

            # Set adversarial and noise_mode
            conf_dict["adversarial"] = adversarial
            conf_dict["noise_mode"] = noise_mode

            # Convert dictionary to OmegaConf object for saving
            conf = OmegaConf.create(conf_dict)
            tags_proc = tags
            tags_proc[1] = tags_proc[1].split("_")[0]
            tags_proc[2] = "baseline" if tags_proc[2] == "None" else tags_proc[2]

            # Add noise_mode to the beginning of the tags
            if noise_mode:
                tags_proc.remove(noise_mode)  # remove noise_mode from its current position
                tags_proc.insert(0, noise_mode)  # insert noise_mode at the beginning

            if adversarial:
                tags_proc.remove('adversarial')  # remove noise_mode from its current position
                tags_proc.insert(0, 'adversarial')  # insert noise_mode at the beginning

            exp_name = "_".join(tags_proc)

            # Save the configuration to a .yaml file
            with open(os.path.join(task_dir, f"{exp_name}.yaml"), "w") as f:
                f.write("# @package _global_ \n")
                OmegaConf.save(config=conf, f=f)
            print(f"Generating experiment: {exp_name}")


if __name__ == "__main__":
    
    # Locate the root directory of the project
    project_root = rootutils.find_root(".", ".project-root")
    print(project_root)

    # Define the directory to save experiment configurations
    exp_dir = os.path.join(project_root, "configs", "experiment")

    # Start generating experiment configurations
    # mse_path = "/home/aladin/latest/resilient_sfl/mse_files/mnli_mses"
    mse_path = "/home/aladin/latest/resilient_sfl/mse_files/barrage_mses"

    main(experiment_dir=exp_dir, 
         adversarial=True, 
         noise_mode="per_round", 
         mse_base_path=mse_path)
    
    main(experiment_dir=exp_dir, 
         adversarial=False, 
         noise_mode="per_round", 
         mse_base_path=mse_path)
    
    main(experiment_dir=exp_dir, 
         adversarial=True, 
         noise_mode="per_batch", 
         mse_base_path=mse_path)
    
    main(experiment_dir=exp_dir, 
         adversarial=False, 
         noise_mode="per_batch", 
         mse_base_path=mse_path)