
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
...
