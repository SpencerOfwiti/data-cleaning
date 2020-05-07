# Exploratory Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) 
![GitHub repo size](https://img.shields.io/github/repo-size/SpencerOfwiti/exploratory-data-analysis.svg)
![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)
[![contributors](https://img.shields.io/github/contributors/SpencerOfwiti/exploratory-data-analysis.svg)](https://github.com/SpencerOfwiti/exploratory-data-analysis/contributors)

An in depth analysis of different data cleaning and exploration techniques and their implementations on real-life datasets.

## Table of contents
* [Build Status](#build-status)
* [Built With](#built-with)
* [Features](#features)
* [Code Example](#code-example)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Contributions](#contributions)
* [Bug / Feature Request](#bug--feature-request)
* [Authors](#authors)
* [License](#license)

## Build Status

[![Build Status](https://travis-ci.com/SpencerOfwiti/exploratory-data-analysis.svg?branch=master)](https://travis-ci.com/SpencerOfwiti/exploratory-data-analysis)

## Built With
* [Python 3.6](https://www.python.org/) - The programming language used.
* [SciKit Learn](https://scikit-learn.org/stable/) - The machine learning library used.
* [Travis CI](https://travis-ci.com/) - CI-CD tool used.

## Features

- Data exploration
- Data Preprocessing
- Data Cleaning

## Code Example

```python
# Preview datatypes for columns
def col_mapping(df):
	df_dtypes = pd.DataFrame(df.dtypes, columns=['dtypes'])
	df_dtypes = df_dtypes.reset_index()
	df_dtypes['name'] = df_dtypes['index']
	df_dtypes = df_dtypes[['name', 'dtypes']]
	df_dtypes['first_value'] = df.loc[0].values
	preview = df_dtypes.merge(data_dictionary, on='name', how='left')
	return preview


preview = col_mapping(data)
```

## Prerequisites

What things you need to install the software and how to install them

* **python 3**

Linux:
```
sudo apt-get install python3.6
```

Windows:

Download from [python.org](https://www.python.org/downloads/windows/) 

Mac OS:
```
brew install python3
```

* **pip**

Linux and Mac OS:
```
pip install -U pip
```

Windows:
```
python -m pip install -U pip
```

## Installation

Clone this repository:
```
git clone https://github.com/SpencerOfwiti/exploratory-data-analysis
```

To set up virtual environment and install dependencies:
```
source setup.sh
```

To run python scripts:
```
python3 src/data/house_prices.py
```

## Contributions

To contribute, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).


## Bug / Feature Request

If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here](https://github.com/SpencerOfwiti/exploratory-data-analysis/issues/new) by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](https://github.com/SpencerOfwiti/exploratory-data-analysis/issues/new). Please include sample queries and their corresponding results.

## Authors

* **[Spencer Ofwiti](https://github.com/SpencerOfwiti)** - *Initial work* 
    
[![github follow](https://img.shields.io/github/followers/SpencerOfwiti?label=Follow_on_GitHub)](https://github.com/SpencerOfwiti)
[![twitter follow](https://img.shields.io/twitter/follow/SpencerOfwiti?style=social)](https://twitter.com/SpencerOfwiti)

See also the list of [contributors](https://github.com/SpencerOfwiti/exploratory-data-analysis/contributors) who participated in this project.

## License

This project is licensed under the MIT license - see the [LICENSE.md](LICENSE.md) file for details
