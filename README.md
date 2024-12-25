## Add your Documantion here the description can be found on [TUWEL](# Project Name

## Overview
This project involves downloading datasets and utilizing a variety of machine learning and data processing libraries to work with these datasets. The project is split into multiple components, including downloading files, processing data, and building models.

## File Descriptions

### `downloadfiles.py`
This script is designed to download a set of files from specified URLs and extract them if they are zip files. It uses the `requests` library to download the files and `tqdm` for progress tracking.

### `npa.ipynb`
This Jupyter Notebook implements NPA algorithm includes code that leverages several machine learning libraries such as `transformers`, `tensorflow`, and `polars` for processing and analyzing datasets. The notebook contains cells to tokenize data, manage paths, and perform various utilities specific to the `ebrec` package. 

The NPA algorithm, or Non-negative Principal Component Analysis, is a method used in data analysis to simplify large datasets while ensuring that the simplified data remains meaningful and easy to interpret. Here's a simple breakdown:

Purpose: The main goal is to reduce the complexity of a large dataset by finding new, smaller sets of data (called components) that still capture the important information from the original data.

Non-negative: The algorithm ensures that all the values in these new components are non-negative, meaning they are zero or positive. This is important in fields like image processing or biology, where negative values don't make sense.

Principal Components: These are the new sets of data that the algorithm finds. They are like summaries of the original data, showing the main patterns or features.

Process: The algorithm looks at the original data, identifies the main features, and creates new components that are easier to work with but still represent the original data well.

### `requirements.txt`
This file lists the dependencies required for the project. It includes specific versions of libraries to ensure compatibility.

