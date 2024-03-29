# pytda

This package makes extensive use of the Python bindings for the Dionysus C++ library (http://mrzv.org/software/dionysus2).

In order to install Dionysus you will need

    * CMake
    * GCC >= 5.4
    * Boost 1.55.

On OSX GCC and CMake and Boost may all be installed with homebrew (https://brew.sh/)

    brew install gcc cmake boost

On Ubuntu

    sudo apt install libboost-all-dev cmake

To build the package and install all requirements run the following from the project's root directory

    pip install -r requirements.txt
    python setup.py build

A modified version of the PyDEC library (https://github.com/hirani/pydec) is included as it is currently not maintained.

Tested on OSX 10.13 and Ubuntu 16.04 with Python 2.7
