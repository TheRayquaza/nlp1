{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-23.11.tar.gz") {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python311Packages.virtualenv
    pkgs.cudaPackages.cudnn
    pkgs.cudaPackages.libcublas
    pkgs.cudaPackages.cuda_nvcc
    pkgs.cudaPackages.cudatoolkit
    #pkgs.cudaPackages.cudatoolkit
    (pkgs.python311.withPackages (ps: with ps; [
      pandas
      numpy
      scipy
      fastapi
      hydra-core
      pydantic
      scikit-learn
      joblib
      nltk
      mlflow
      rich
      boto3
      jupyter
      seaborn
      matplotlib
      spacy
      torch
      torchmetrics
      notebook
      rapidfuzz
      pip
      setuptools
      gensim
      transformers
      tqdm
      optuna
      pycuda
    ]))
  ];

   profile = ''
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.cudaPackages.libcublas}/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudnn}/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.cudaPackages.cudatoolkit}/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
  '';
 
  shellHook = ''
    VENV_DIR=".venv-tf"

    if [ ! -d "$VENV_DIR" ]; then
      echo "Creating virtualenv for TensorFlow in $VENV_DIR"
      python -m venv "$VENV_DIR"
      source "$VENV_DIR/bin/activate"
      pip install --upgrade pip
      pip install tensorflow[and-cuda] ipykernel
      python -m ipykernel install --user --name=tf-venv --display-name "Python (TensorFlow venv)"
    else
      echo "Using existing virtualenv in $VENV_DIR"
      source "$VENV_DIR/bin/activate"
    fi

    echo "To use TensorFlow, select the 'Python (TensorFlow venv)' kernel in Jupyter."
    jupyter notebook
  '';
}

