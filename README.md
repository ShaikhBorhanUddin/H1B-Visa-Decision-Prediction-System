# H1B Visa Approval Prediction for Employer Decision Support (project ongoing) 
<p align="left">
  <!-- Core -->
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white" alt="Made with Colab">
  <img src="https://img.shields.io/badge/Language-Python-green?logo=python" alt="Language: Python">
  <img src="https://img.shields.io/badge/💻Dev%20Environment-VS%20Code-blue?logo=visualstudiocode">

  <!-- License & Issues -->
  <img src="https://img.shields.io/badge/⚖️%20License-MIT-red" alt="License">
  <img src="https://img.shields.io/badge/🐞%20Issues-None-green" alt="Issues">

  <!-- Repo Stats -->
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/H1B-Visa-Approval-Prediction-for-Employer-Decision-Support?logo=github" />
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/H1B-Visa-Approval-Prediction-for-Employer-Decision-Support" alt="Last Commit">

  <!-- Models -->
  <img src="https://img.shields.io/badge/🤖%20Models-XGBoost%20|%20LightGBM%20|%20Random%20Forest-red" alt="Models">

  <!-- Dataset -->
  <img src="https://img.shields.io/badge/🗂️Dataset-Kaggle-blueviolet" alt="Dataset">
  
  <!-- Explainability -->
  <img src="https://img.shields.io/badge/🧠Explainability-SHAP-orange" alt="Explainability">

  <!-- Visualization -->
  <img src="https://img.shields.io/badge/📊%20Visualization-Matplotlib%20|%20Seaborn-green" alt="Visualization">

  <!-- Deployment -->
  <img src="https://img.shields.io/badge/🌐Deployment-React | AWS EC2-orange" alt="Deployment">

  <!-- API / Backend -->
  <img src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi" alt="FastAPI">

  <!-- Frontend -->
  <img src="https://img.shields.io/badge/Frontend-React | HTML5 | CSS3-61DAFB?logo=react&logoColor=black" alt="React">

  <!-- Containerization -->
  <img src="https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white" alt="Docker">

  <!-- Runtime -->
  <img src="https://img.shields.io/badge/⚙️Runtime-GPU%20Training%20|%20CPU%20Inference-blue" alt="Runtime"> 
  
  <!-- DevOps -->
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Git">
  <img src="https://img.shields.io/badge/Code-GitHub-red?logo=github" alt="GitHub">
  
  <!-- Social -->
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/H1B-Visa-Approval-Prediction-for-Employer-Decision-Support?style=social" alt="Forks">

  <!-- Status -->
  <img src="https://img.shields.io/badge/🏁Project-In%20Development-yellow" alt="Status">
</p> 

![dashboard](assets/title.png) 

## Overview 
This project presents an end-to-end machine learning system for H-1B visa approval prediction designed to support employer decision-making using approximately 3.5 million visa application records from 2020–2024. Multiple machine learning models including XGBoost, LightGBM, Random Forest, and ElasticNet are trained and evaluated using performance metrics and SHAP-based explainability to identify the most influential approval drivers. The final selected model is deployed through a Gradio interface on Hugging Face for real-time inference, with future plans for scalable cloud deployment using FastAPI, Docker, AWS, and a React frontend for production-grade serving and interactive user experience. 

## Use Cases 
The H-1B visa process is highly selective, time-consuming, and financially expensive for sponsoring employers, involving legal fees, compliance documentation, prevailing wage requirements, attorney support, and operational delays that may still result in petition denial. A rejected application can lead to significant loss of time, recruitment resources, project delays, and reputational impact for both the applicant and the sponsoring organization. This project is designed as an employer-focused decision support system that helps companies assess the likelihood of H-1B visa approval before formally initiating the sponsorship process. By analyzing factors such as offered wage, prevailing wage alignment, occupation category, employer compliance indicators, worksite characteristics, petition structure, and historical filing patterns, the system provides predictive insights that can assist employers in preliminary candidate screening, sponsorship feasibility analysis, and risk-aware hiring decisions. The application can also help organizations identify cases that may require stronger legal or attorney support, additional documentation preparation, or compensation adjustments before filing, enabling more informed and strategically optimized H-1B sponsorship decisions. 

## Project Workflow 
![dashboard](assets/wf.png) 

## Dataset 
The [Dataset](https://www.kaggle.com/datasets/zongaobian/h1b-lca-disclosure-data-2020-2024) is sourced from Kaggle. It provides a comprehensive record of Labor Condition Application (LCA) disclosures for H1B visa petitions filed with the U.S. Department of Labor (DOL) from 2020 to 2024. The H1B visa is a non-immigrant visa that allows U.S. companies to employ foreign workers in specialty occupations requiring theoretical or technical expertise. These roles typically include fields such as IT, engineering, finance, healthcare, and more. The H1B program is critical for addressing skill gaps in the U.S. workforce and supporting economic growth. 

More recent [data](https://www.kaggle.com/datasets/saurabhbadole/u-s-immigration-and-wage-disclosure-data-updated) of 2025 (upto June) was also considered. 

[SOC Code](https://www.bls.gov/oes/2023/may/oes_stru.htm) 

[NAICS](https://www.census.gov/naics/) 

[Up to date SOC](https://www.onetonline.org/find/all) 

A [second H1B dataset](https://www.kaggle.com/datasets/kanishk307/h1b-applications-dataset-from-2015-to-2019/data) of 2015-2019 records was initially also cosidered. However, combining the 2015–2019 H1B dataset with the 2020–2024 dataset would increase the total number of records, it may reduce overall data consistency and model reliability. The older dataset lacks several important petition-type features such as NEW_EMPLOYMENT, CHANGE_EMPLOYER, AMENDED_PETITION, and NEW_CONCURRENT_EMPLOYMENT, which are highly relevant for employer-side H1B decision support. Merging both datasets would therefore introduce large amounts of structurally missing data and potentially distort model learning. In addition, the SOC occupation classification system changed around 2018–2019, meaning older occupation codes may not align properly with newer records. The H1B policy environment, labor market conditions, and visa scrutiny standards also changed significantly between 2015 and 2024, making the older data less representative of the current sponsorship landscape. Since the 2020–2024 dataset already contains approximately 3.5 million records with richer and more modern features, it is sufficient for building a scalable, deployment-ready machine learning model without introducing schema inconsistencies and temporal drift from older data.


## Folder structure 

```bash
project/
│
├── app/
│   ├── app.py
│   │
│   ├── templates/
│   │   └── index.html
│   │
│   └── static/
│       └── style.css
│
├── assets/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│   └── 05_inference.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
```

## Data Cleaning 
Keep SOC_CODE JOB_TITLE FULL_TIME_POSITION PREVAILING_WAGE WAGE_RATE_OF_PAY WAGE_UNIT_OF_PAY PW_LEVEL WORKSITE_STATE WORKSITE_CITY NAICS_CODE H1B_DEPENDENT WILLFUL_VIOLATOR YEAR 

## Feature Engineering 
calculate job duration from BEGIN_DATE and END_DATE. 

## EDA 

## Model Training 

## Test Results 

## Result Visualization and Explanations 

## Deployment 

## Explainability 
![dashboard](assets/Top_5_SHAP_Drivers.png) 

## Limitations 

## Tools and Technology Used 

## Licence 

## Contact 


