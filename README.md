# The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators

**Abstract:** A core challenge in scientific machine learning, and scientific computing more generally, is modeling continuous phenomena which (in practice) are represented discretely. Machine-learned operators (MLOs) have been introduced as a means to achieve this modeling goal, as this class of architecture can perform inference at arbitrary resolution. In this work, we evaluate whether this architectural innovation is sufficient to perform “zero-shot super-resolution,” namely to enable a model to serve inference on higher-resolution data than that on which it was originally trained. We comprehensively evaluate both zero-shot sub-resolution and super-resolution (i.e., multi-resolution) inference in MLOs. We decouple multi-resolution inference into two key behaviors: 1) extrapolation to varying frequency information; and 2) interpolating across varying resolutions. We empirically demonstrate that MLOs fail to do both of these tasks in a zero-shot manner. Consequently, we find MLOs are not able to perform accurate inference at resolutions different from those on which they were trained, and instead they are brittle and susceptible to aliasing. To address these failure modes, we propose a simple, computationally-efficient, and data-driven multi-resolution training protocol that overcomes aliasing and that provides robust multi-resolution generalization.

<img width="682" height="370" alt="image" src="https://github.com/user-attachments/assets/09a29bce-43e6-47d8-a2b0-91e0917dd97b" />

*Aliasing in zero-shot super-resolution. Model trained on resolution 16 data, and evaluated at varying resolutions: 16, 32, 64, 128. Top Row: Sample prediction for Darcy flow; notice
striation artifacts at resolution 128. Middle Row: Average test set 2D energy spectrum of label and model prediction. Bottom Row: Average residual spectrum normalized by label spectrum.*

We give a high-level overview of the code structure in this repository below. More detailed READMEs can be found in every subdirectory with pointers to any external repos we utilized or took inspiration from. If there are any questions or concerns, please feel free to open a github issue or email  `sakarvadia@uchicago.edu`.

# Training Models on PDE Data
- [`main.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/operator_aliasing/main.py) allows you to run a single experiment w/ custom experimental configurations. Do `python main.py --help` for details. Running `python main.py` will train an FNO on a darcy flow dataset from PDEBench.
- [`train`](https://github.com/msakarvadia/operator_aliasing/tree/main/operator_aliasing/train) contains training utility codes

# Data
- [`data`](https://github.com/msakarvadia/operator_aliasing/tree/main/operator_aliasing/data) data downloading + processing scritps for Darcy Flow, Burgers, and Incompressible Navier Stokes data.

# Models
- [`models`](https://github.com/msakarvadia/operator_aliasing/tree/main/operator_aliasing/models) contains model definitions and the utilities to instantiate them


# Experimental Configuration/Launch Scripts

All experimental codes are in [`experiments`](https://github.com/msakarvadia/operator_aliasing/tree/main/experiments). All experiments are parallelized using the [`parsl`](https://parsl-project.org/) framework.
- [`parsl_setup.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/parsl_setup.py): set up Parsl configuration specific to your computing envionrment here. The default settings are specific to the [`Perlmutter`](https://docs.nersc.gov/systems/perlmutter/architecture/) machine.
- [`get_train_args.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/get_train_args.py): script to compile configures for each experiment (iterates through all variation of the experiment in the paper)
- [`training.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/training.py): script to run a specific set of expreiments.
  - By default `python training.py` will run the information extrapolation/resolution interpolation experiments from the paper.
  - Toggle `python training.py --help` to see all avaliable experiments.

# Figures

- [figures.ipynb](https://github.com/msakarvadia/operator_aliasing/blob/train/notebooks/figures.ipynb) is used to process experimental results and generate figures.


## Installation

For local development:
```
git clone https://github.com/msakarvadia/operator_aliasing.git # swap url for clone via SSH
cd operator_aliasing
conda create -p env python==3.10 # we choose conda to manage env, but venv is another option
conda activate env
pip install -e .[dev]
pre-commit install
```

## Citation

Please cite this work as:

```bibtex
@article{sakarvadia2025false,
      title={The False Promise of Zero-Shot Super-Resolution in Machine-Learned Operators}, 
      author={Mansi Sakarvadia and Kareem Hegazy and Amin Totounferoush and Kyle Chard and Yaoqing Yang and Ian Foster and Michael W. Mahoney},
      year={2025},
      eprint={...},
      url={...}, 
}
```
