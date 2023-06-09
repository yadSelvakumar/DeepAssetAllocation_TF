# Deep Asset Allocation Tensorflow

## Setting Up

---

### Conda Setup

```bash
bash setup/tflow.sh
```

### Project dependencies

Only if you didn't need the conda environment setup

```bash
pip install -r setup/requirements.txt
```

### Check Tensorflow Works Correctly

You can either use the tflow.sh setup, or run the following:

> Warning: You'll need make to run this commands
> If you don't want to install it with apt, just run the commands inside setup/Makefile

```bash
make -C setup tfcpu # To test CPU
make -C setup tfgpu # To test GPU
```

## Using the project

---

### Running the training

To run with default parameters:

```bash
python main.py
```

You can see the available parameters with:

```bash
python main.py --help
```

### Running the tests

If you installed the project dependencies, you can run the tests with just:

```bash
pytest
```

> If you are in vscode or pycharm, you can run and check the tests from the IDE
