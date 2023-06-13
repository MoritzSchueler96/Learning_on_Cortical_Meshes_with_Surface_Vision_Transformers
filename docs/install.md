## Installation

We provide some advice for installation all dependencies with conda.

### Prepare environment

1. create a conda environement

```
conda create -n sit python=3.7
```

2. activate the environment

```
conda activate sit
```

3. install requirements

```
pip install -r requirements.txt
```

4. install ray tune

```
pip install -U "ray[tune]" torch torchvision pytorch-lightning
```

5. install own lib, cd into code/

```
pip install -e .
```
