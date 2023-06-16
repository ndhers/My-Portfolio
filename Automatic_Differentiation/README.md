# cs107-FinalProject 

[![CircleCI](https://circleci.com/gh/cs107-jnrw/cs107-FinalProject/tree/master.svg?style=svg&circle-token=b6d211291a7a8ef00d4ff35ce60134c8ca0e6ed3)](https://circleci.com/gh/cs107-jnrw/cs107-FinalProject/tree/master) [![codecov](https://codecov.io/gh/cs107-jnrw/cs107-FinalProject/branch/master/graph/badge.svg?token=XZ5JF8H40H)](https://codecov.io/gh/cs107-jnrw/cs107-FinalProject)

**Group member**: Joslyn Fu, Nicolas Dhers, Rui Cheng, William Tong

## Installation
Install from PyPI:

```sh
pip install AutoDiff-jnrw
```

Alternatively, install directly from this repo:

Clone the repo
```sh
git clone https://github.com/cs107-jnrw/cs107-FinalProject.git
```

Install using `pip`
```sh
cd cs107-FinalProject
pip install .
```

## Testing
This project uses `pytest` for testing. To run the tests suite, simply call

```sh
pytest
```

To generate browsable coverage reports
```sh
coverage run -m pytest
coverage html
```

Or alternatively, view the coverage reports from `CodeCov` by clicking on the badge above.

## About the Project
This project is a work-in-progress. To learn more about usage and status, please see the documents in `docs`. Below is a brief description of their content

- milestone1: This document is for planning purpose only. And due to incapability of rendering LATeX at github web page. We have further included a separate pdf document as advised in the same folder for reading purpose.

- milestone2_progress: This document is for planning purpose only. The document briefed tasks each member of the group has completed since the submission of milestone1 and allocated tasks for each group member for milestone2. The allocated tasks are indicative only and members are expected to cooperate and interact on each task along the way.

- milestone2: Updated version of mileston1 documentation. Please refer to this document for latest updates including forward mode implementation, scalar functions and test process.

- documentation: Final version of package documentation. Please refer to this document for complete information on our library including background, installation, testing, software organization, implementation, extension, broader impact and inclusivity statement and future work. 
