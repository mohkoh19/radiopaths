# Radiopaths

In this repository, we release the code for our paper "Radiopaths: Deep Multimodal Analysis on Chest Radiographs". It contains:
- A Jupyter Notebook for generating the datasets that we uses (provided you have credentialed access to PhysioNet)
- Our unimodal and multimodal models in `src/models`
- Classes for loading the final dataset in `src/datasets`
- Code for training and testing the models

We use PyTorch and Lightning for developing the models, dataset classes, and the overall optimization pipeline. We use [Weights and Biases](https://wandb.ai/site) for logging and expect you to have an account on WandB and to be logged in on your system. More info can be found [here](https://docs.wandb.ai/quickstart).

## Requirements

- Python 3
- Conda
- Credentialed Access to PhysioNet

## Getting Started

1. To get started first download the repository and create the conda environment. We use Poetry for dependency management. Poetry will be installed in the environment. We then install all the dependencies.
   ``` 
   git clone git@github.com:mohkoh19/radiopaths.git
   cd radiopths
   conda env create -f environment.yml
   poetry install
   ```
2. Login to your WandB account
   ```
   wandb login
3. Generate the dataset by running the `generate_dataset.ipynb` notebook in the `jupyter` folder. The requirements and the specific steps are described in the notebook. A file named `mimic_cxr_mv.pkl` will be generated containing the dataset and the downscaled JPEG version of MIMIC-CXR will be downloaded.
4. If you use VS Code, you can use the `launch.json`  file in `.vscode/` to run `train.py `and `test.py` in the Run and Debug view. Here we also saved the hyperparameters for our best models for each experiment. Please note that you need to fill arguements, which expect local paths from you (e.g., `df_file` the path to the dataset). 
Otherwise you can run the files as normal python scripts. You can use 
    ```
    python <train/test>.py -h 
    ```

    to display all the possible args both from the Lightning trainer and from our modules. If a module has configurable arguements/hyperparameters, they will be listed in its `add_model_specific_args` method and in the parameters of it's constructor. You can use the DocStrings to understand the functionality of the specific parameters.
    
    
## Acknowledgement
In case you use code from our repository, we would be grateful if you cite us:
```
@INPROCEEDINGS{kohankhaki2022radiopaths,
  author={Kohankhaki, Mohammad and Ayad, Ahmad and Barhoush, Mahdi and Leibe, Bastian and Schmeink, Anke},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)}, 
  title={Radiopaths: Deep Multimodal Analysis on Chest Radiographs}, 
  year={2022},
  volume={},
  number={},
  pages={3613-3621},
  doi={10.1109/BigData55660.2022.10020356}
}
```
