### Hi Welcome to BugRMSys

# BugRMSys - README

BugRMSys is a system for managing and analyzing bug reports in software applications. This README provides detailed instructions on how to reproduce results using the Python scripts located in the `src` directory.

## Project Structure

The `src` directory contains several Python scripts, each serving a specific purpose in the project. Here is a brief overview of each script:

- `Constants.py`: Contains constant values used across other scripts.
- `ConstantsRQ4.py`: Specific constants for research question 4.
- `DataLoader.py`: Script for loading and processing data.
- `EmbeddingCalculation.py`: Calculates embeddings for the data.
- `GithubIssueGet.py`: Retrieves issues from GitHub.
- `MatchTest.py`: Tests the matching algorithm.
- `ModelLoader.py`: Loads the machine learning model.
- `RQ1.py`, `RQ2.py`, `RQ2_1_2.py`, `RQ2b.py`: Scripts related to specific research questions.
- `ReviewGet.py`: Retrieves reviews for analysis.

## Setup and Installation

Before running the scripts, ensure that you have Python installed on your system along with the required libraries. You might need libraries such as `pandas`, `numpy`, `matplotlib`, and others depending on the script requirements.

## Usage Instructions

### `Constants.py`

- This file contains constants used in other scripts.
- **Usage**: Import this file in other scripts where these constants are needed.

### `ConstantsRQ4.py`

- Constants specific to research question 4.
- **Usage**: Import in scripts dealing with RQ4.

### `DataLoader.py`

- Used for loading and processing data.
- **Run the script**: `python DataLoader.py`
- Ensure the necessary data files are available in the specified directories.

### `EmbeddingCalculation.py`

- Calculates embeddings for the data.
- **Run**: `python EmbeddingCalculation.py`
- Requires pre-processed data from `DataLoader.py`.

### `GithubIssueGet.py`

- Retrieves issues from GitHub repositories.
- **Run**: `python GithubIssueGet.py`
- Set up necessary API keys and target repositories.

### `MatchTest.py`

- Tests the matching algorithm on the data.
- **Run**: `python MatchTest.py`
- Requires output data from other preprocessing scripts.

### `ModelLoader.py`

- Loads machine learning models for analysis.
- **Run**: `python ModelLoader.py`
- Ensure model files are correctly placed in the designated directories.

### `RQ1.py`, `RQ2.py`, `RQ2_1_2.py`, `RQ2b.py`

- Scripts for specific research questions.
- **Run each script with**: `python [script_name].py`
- Each script may require specific input data or parameters; ensure they are correctly set.

### `ReviewGet.py`

- Retrieves and processes reviews.
- **Run**: `python ReviewGet.py`
- Requires internet connection and access to review sources.

## Contributing

Contributions to BugRMSys are welcome. Please follow the standard procedures for contributing to Python projects, including writing clean, well-documented code and submitting pull requests for review.

## License

(Include license information here)


<!--
**BugRMSys/BugRMSys** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

## Versions of apps we used, device information, and system information
* Reproducing device: Xiaomi 11 Ultra
* System: MIUI 13.0.9.0
* Android version: 12
* Version of Wire: 3.80.23
* Version of Signal: 5.32.15
* Version of Firefox: 97.3.0
* Version of Brave: 1.35.103
* Version of Nextcloud: 
* Version of Owncloud:
* GPU: Tesla V100 32GB
* CPU: 40  Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz

## data availability
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7520604.svg)](https://doi.org/10.5281/zenodo.7520604)
