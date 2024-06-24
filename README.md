# flim_granules_analysis

## Description
This is a collection of integrative tools designed to analyze FLIM (Fluorescence Lifetime Imaging Microscopy) images of granules (vesicles) in cells. It provides capabilities for granule segmentation (in one channel or colocalizing from two channels) to derive average lifetimes from the objects. These lifetimes can be used to predict the pH of granules (vesicles), based on a calibration curve of the reporter.  Additionally, it includes batch processing capabilities for analyzing multiple images.

### Inputs
The suite supports exponentially fitted images in `.tiff` format (performed externally) or raw data in `.ptu` format (only for phasor analysis with the support of the [napari-flim-phasor-plotter plugin](https://github.com/zoccoler/napari-flim-phasor-plotter).

## Features
- **Segmentation**: Segment granules (vesicles) in single-channel images or colocalized objects in two-channel images.
- **Lifetime Calculation**: Derive the average lifetimes of segmented objects.
- **Phasor Analysis**: Perform phasor analysis on raw `.ptu` files with support from the [napari-flim-phasor-plotter plugin](https://github.com/zoccoler/napari-flim-phasor-plotter).
- **pH Prediction**: Optionally predict the pH of objects based on a calibration curve. The model built using the Henderson-Hasselbalch equation.
- **Batch Processing**: Process multiple images from folders, providing name IDs as columns in the output `.csv`  file 

### Examples of Use

Each analysis option is exemplified in a separate notebook with detailed description. The functionality allows you to select analysis for single-channel or colocalized objects and provides the choice of FLIM analysis from: Fast FLIM (non-weighted lifetimes), fitted and weighted lifetimes, or load raw data in `.ptu` and analyze with Phasor with the [napari-flim-phasor-plotter plugin](https://github.com/zoccoler/napari-flim-phasor-plotter).

- **Analize fitted/ Fast FLIM images**: 
  - [Single Channel Fast FLIM Analysis](notebooks/Image_FastFLIM.ipynb)
  - [Single Channel Fitted FLIM Analysis](notebooks/Image_FitFLIM.ipynb)
  - [Double-Positive Fast FLIM Analysis](notebooks/Image_coloc_FastFLIM.ipynb)
  - [Double-Positive Fitted FLIM Colocalization Analysis](notebooks/Image_coloc_FitFLIM.ipynb)

- **Phasor Analysis**:
  - [Single Channel Phasor Analysis](notebooks/Image_Phasor.ipynb)
  - [Double-Positive Phasor Analysis](notebooks/Image_Phasor_coloc.ipynb)

- **Folder Processing**: Automatically read pairs of fluorescent intensity and fitted images from a folder, and select the analysis option.
  - [Folder Processing](notebooks/Folder_process.ipynb)

- **Batch Processing**: Batch process multiple folders at the same time, where folder names serve as specific identifiers in the output `.csv` file.
  - [Batch Processing Subfolders](notebooks/Batch_Processing_subfolders.ipynb)
  - [Batch Processing with Name Tags](notebooks/Batch_with_name_tags.ipynb)

- **pH Calibration and Prediction**:
  - [pH Calibration](notebooks/pH_calibration.ipynb): Build a calibration curve and model using the Henderson-Hasselbalch equation.
  - [Batch pH Processing](notebooks/Batch_pH_Processing.ipynb): Batch process imaged for pH calibration.
  - [Predict pH from Image Data](notebooks/Get_ph_from_imagedata.ipynb): Use the model to predict the pH from imagimg data of previously segmented objects.


### Requirements
The code relies on different libraries, packages, and plugins to run successfully. The list of dependencies is provided in the [requirements.txt](link/to/your/requirements.txt) file.

## Installation

Create a conda environment named `myenv` with Python version 3.9:
mamba create --name myenv python=3.9

Activate the newly created environment:
conda activate myenv

Install Required Dependencies
pip install -r requirements.txt

This command will install all necessary libraries, packages, and plugins listed in the `requirements.txt` file, ensuring that the `flim_granules_analysis` tool runs successfully.


