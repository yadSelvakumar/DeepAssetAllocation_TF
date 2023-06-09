all: update condadownload condasource condaenv condaforge condapath tensorflow tfcpu tfgpu pipinstall

condadownload:
	curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh

condasource:
	source ~/.bashrc && conda -V

update:
	sudo apt update && sudo apt upgrade
	sudo ubuntu-drivers autoinstall

condaenv:
	conda create --name tf python=3.10
	conda activate tf

condaforge:
	conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

condapath:
	mkdir -p $CONDA_PREFIX/etc/conda/activate.d
	echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

tensorflow:
	pip install --upgrade pip
	pip install tensorflow=2.9.2

tfcpu:
	python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

tfgpu:
	nvidia-smi
	python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

pipinstall:
	pip install -r requeriments.txt
