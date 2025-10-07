All experiments are parallelized using the [`parsl`](https://parsl-project.org/) framework.
- [`parsl_setup.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/parsl_setup.py): set up parl configuration specific to your computing envionrment here. The default settings are specific to the [`Perlmutter`](https://docs.nersc.gov/systems/perlmutter/architecture/) machine.
- [`sample_experiment.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/sample_experiment.py): "Hello World" experiment, useful for testing parallelizaiton of experimental infrastructure
- [`get_train_args.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/get_train_args.py): script to compile configures for each experiment (iterates through all variation of the experiment in the paper)
- [`training.py`](https://github.com/msakarvadia/operator_aliasing/blob/main/experiments/training.py): script to run a specific set of expreiments.
  - By default `python training.py` will run the information extrapolation/resolution interpolation experiments from the paper.
  - Toggle `python training.py --help` to see all avaliable experiments.
