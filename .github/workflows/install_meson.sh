#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo
    echo "install_meson.sh /installation/path {macOS, Linux}"
    echo
    exit 1
fi
install_path=$1
runner_os=$2

# Beginning with v1.6.0 meson can automatically find OpenMPI and MPICH
if   [ "$runner_os" = "macOS" ]; then
    # Homebrew already has v1.6.0 available.
    brew install meson
elif [ "$runner_os" = "Linux" ]; then
    # Meson versions available through Ubuntu package installation can be quite
    # out-of-date.
    venv_path=$install_path/local/venv
    meson_venv=$venv_path/meson
    local_bin=$install_path/local/bin

    sudo apt-get update
    sudo apt-get -y install ninja-build

    echo " "
    mkdir -p $venv_path
    mkdir -p $local_bin
    
    python -m venv $meson_venv
    . $meson_venv/bin/activate
    which python
    which pip
    python -m pip install --upgrade pip setuptools
    python -m pip install meson>=1.6.0
    echo " "
    python -m pip list
    echo " "
    deactivate
    
    # Install just meson command in path for all subsequent steps in job
    ln -s $meson_venv/bin/meson $local_bin
    echo "$local_bin" >> "$GITHUB_PATH"
    echo " "
else
    echo
    echo "Invalid runner OS $runner_os"
    echo
    exit 1
fi
