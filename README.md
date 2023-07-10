# Quality & Usability Project 3: Text Classification based on Sustainable Development Goals (SDGs)

---
## Table of Contents
1. [Project Description](#project-description)
2. [Datasets](#datasets)
3. [Model](#model)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)
8. [Acknowledgements](#acknowledgements)
9. [Appendix](#appendix)
10. [References](#references)
11. [Project Status](#project-status)
12. [To Do](#to-do)

---
##  [Project Description](#project-description)

### Background

The Sustainable Development Goals (SDGs) are a collection of 17 global goals set by the United Nations General Assembly in 2015 for the year 2030. The SDGs are part of Resolution 70/1 of the United Nations General Assembly, the 2030 Agenda. The SDGs build on the principles agreed upon in Resolution A/RES/66/288, entitled "The Future We Want". This resolution was a broad intergovernmental agreement that acted as the precursor for the SDGs. The goals are broad-based and interdependent. The 17 sustainable development goals each have a list of targets that are measured with indicators. The total number of targets is 169. Below is an image of the 17 SDGs:


<img src="./sdg_un_goals_img.png" width="300">


### Project Goal

The goal of this project is to classify text data into the 17 SDGs. The project will use 4 different datasets consisting of 65938 labeled sentences. The dataset is available on Huggingface. The project will use Natural Language Processing (NLP) techniques to classify the text data into the 17 SDGs. The project will use a transformer architecture combined with a classification layer to classify the textual data. Furthermore, the project will utilize a stratified K-fold approach along with the F1 score to evaluate the performance of the classification model and guarantee consistency throughout the unlabeled dataset.

### Project Motivation

The motivation for this project is to use NLP techniques to classify text data into the 17 SDGs. Moreover, the project will develop a evaluation pipeline that will be applied on 50 scraped published sustainability reports from various goods/services companies.

---

## [Datasets](#datasets)
The project will use 4 different datasets consisting of 65938 labeled sentences. The dataset is available on Huggingface. The datasets are as follows:

- original_data.csv: 2567 labeled sentences (supplied by the Quality & Usability supervision team)
- politics.csv: 22977 labeled sentences (supplied by the Quality & Usability supervision team)
- targets.csv: 332 labeled sentences (supplied by the Quality & Usability supervision team)
- osdg_data.csv: 40062 labeled sentences (The [OSDG Community Dataset (OSDG-CD)](https://zenodo.org/record/8107038)) 

The entire dataset is available on Huggingface: [amay01/Quality_and_usability_SDG_dataset](https://huggingface.co/datasets/amay01/Quality_and_usability_SDG_dataset)

### Dataset Distribution
#### Single Label Distribution

<img src="./Dataset_Distribution/single_dist.png" width="300">

#### Merged Label Distribution

<img src="./Dataset_Distribution/merged_dist.png" width="300">

---

## [Model](#model)




## [Usage](#usage)

### Environment & Dependencies Setup:

```bash
conda create -n sdg_project python=3.10
conda activate sdg_project
pip install -r requirements.txt
```