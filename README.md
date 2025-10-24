# Humanitarian LLM

## Tech

 Humanitarian LLM uses a number of open source projects to work properly:
- [miniconda] - Miniconda is a free minimal installer for conda.
- [python3.8] - Base Programming Language.
- [higgingace] - Hugging Face is a machine learning (ML) and data science platform and community that helps users build, deploy and train machine learning models.


## Install
```bat
> conda env create -f environment.yml
> conda activate llm
```

## Use

Before the use of the notebooks is mandatory to set your own hugginface ID in config.*.yml files. 


## 📖 Citation

If you use this repository, please cite:

```bibtex
@article{10.1145/3736160,
author = {S\'{a}nchez, Cinthia and Abeliuk, Andr\'{e}s and Poblete, Barbara},
title = {Large Language Models in Crisis Informatics for Zero and Few-Shot Classification},
year = {2025},
issue_date = {November 2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {19},
number = {4},
issn = {1559-1131},
url = {https://doi.org/10.1145/3736160},
doi = {10.1145/3736160},
journal = {ACM Trans. Web},
month = oct,
articleno = {45},
numpages = {25},
keywords = {Instruction fine-tuning, humanitarian information, text classification}
}
