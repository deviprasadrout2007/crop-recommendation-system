# 🌱 Crop Recommendation System

A Streamlit machine learning application that recommends crops using soil and climate data.

## Features

- Dataset upload and automatic dataset loading
- Data cleaning and validation
- Exploratory Data Analysis (EDA)
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- Stratified 5-fold cross-validation
- Test accuracy comparison
- Classification report
- Confusion matrix
- Native and permutation feature importance
- Interactive crop prediction interface

## Required Dataset Columns

```text
n,p,k,temperature,humidity,ph,rainfall,label
```

The app also accepts `crop` instead of `label` as the target column.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run on Replit

Use the following command in the Shell:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 5000
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to GitHub.
2. Open https://share.streamlit.io/
3. Select your repository.
4. Choose `app.py` as the main file.
5. Deploy.

## Important Note

The model's recommendation is based on the training dataset. It should not be treated as professional agricultural advice or a guarantee of crop success.
