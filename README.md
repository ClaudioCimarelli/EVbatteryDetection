# Battery Disassembly Vision System

## Overview

This project aims to automate the disassembly of batteries using a computer vision system. It includes object detection and pose estimation modules designed to identify and manipulate battery components accurately.

## Installation

1. Clone this repository to your local machine,

2. Create conda env and activate it:

        conda create -n env pip python=3.11
        conda activate env

3. Navigate to the project directory and install the package:

        pip install -e .

## Running the Tests

To run the unit tests, navigate to the project root and execute:

        python -m unittest discover tests

or

        pytest


## Project Structure

    .
    ├── README.md
    ├── requirements.txt            # List of dependencies to install using pip
    ├── setup.py                    # Setup script for installing the project
    ├── .gitignore                  # Specifies intentionally untracked files to ignore
    ├── data/
    │   └── images/              # Data that has been manipulated for training/testing
    │   ├────train/              # Train Data
    │   └────test/              # Test Data
    │
    ├── plots_and_logs/       #Output for plots and logs
    │   └── plot.png              # Image of a plot
    │
    ├── notebooks_and_scripts/     # Jupyter notebooks for exploration and presentation
    │   └── detector.py     # Script for simple detection
    │
    ├── src/                        # Source code for the project
    │   ├── __init__.py             # Makes src a Python module
    │   ├── main.py                 # Entry point of the program
    │   ├── vision_system/          # Module for the vision system functionality
    │   │   ├── __init__.py
    │   │   ├── detector.py         # Object detection code
    │   │   └── estimator.py        # Pose estimation code
    │   └── utils/              # Helper functions and utilities
    │       ├── __init__.py
    │       └── data_loader.py      # Utility to load and preprocess data
    │
    └── tests/                      # Unit and integration tests
        ├── __init__.py
        ├── test_detector.py
        ├── test_estimator.py
        └── test_data_loader.py
