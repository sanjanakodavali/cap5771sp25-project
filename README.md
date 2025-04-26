# CAP5771 SP25 Project README

> **Personalized Medication Recommendation System**

---

## 1. Project Overview and Goals
The **Personalized Medication Recommendation System** is an AI-driven pipeline designed to ingest, preprocess, and analyze large-scale prescription and patient datasets to generate tailored drug recommendations. Our aims are to:

1. **Ingest & Preprocess**: Clean and consolidate heterogeneous datasets (patient records, prescription logs, drug interactions).  
2. **Feature Engineering & Selection**: Create and select high-impact features (sentiment scores, interaction counts, average ratings).  
3. **Modeling**: Train and evaluate multiple classifiers (Random Forest, SVM, Logistic Regression, XGBoost) using stratified sampling and reproducible splits.  
4. **Interpretation**: Visualize model outputs and feature importances via SHAP and correlation analyses.  
5. **Deployment**: Deliver an interactive Streamlit dashboard for clinicians to explore and validate recommendations.  

---

## 2. Milestone 2 Highlights (Feature Engineering & Modeling)

During **Milestone 2**, we focused on advanced data transformations and baseline model evaluation:

- **Custom Features**:  
  - Average Drug Rating (per drug)  
  - Sentiment Score (TextBlob polarity on patient reviews)  
  - Review Length & Complexity metrics  
  - Drug Interaction Count aggregated from interaction database  

- **Encoding Strategies**:  
  - Label Encoding for ordinal severity levels  
  - One-Hot Encoding for nominal fields (drug class, manufacturer, condition)  

- **Feature Selection**:  
  - Impurity-based importances from Random Forest  
  - Correlation matrix to detect multicollinearity  

- **Model Training**:  
  - 80:20 stratified train-test split with fixed random seed  
  - Evaluated Random Forest, SVM, Logistic Regression, XGBoost  
  - Key metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
  - **Results**: Random Forest achieved highest F1 and AUC; XGBoost closely matched; SVM and Logistic Regression showed trade-offs between precision and recall  

---

## 3. Technology Stack
- **Language**: Python 3.x  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, WordCloud, TextBlob, Scikit-learn, XGBoost, SHAP  
- **Tools**: Jupyter Notebook, Google Colab, Git & GitHub, Streamlit  

---

## 4. Repository Structure
```bash
cap5771sp25-project/
├── Data/                     # Raw & processed data
│   ├── data_access_info.txt  # How to obtain full datasets
│   ├── drug200.csv           # Sample interaction data
│   ├── medicine_dataset.csv.zip
│   └── medicine_prescription_records.csv.zip
│
├── Scripts/                  # Code & notebooks
│   ├── Milestone1.py         # Preprocessing & EDA
│   ├── Milestone2.py         # Feature engineering & modeling (this milestone)
│   ├── Milestone3.py         # Evaluation & interpretation
│   ├── app.py                # Streamlit dashboard
│ 
├── Report/                   # Written reports
│   ├── Milestone1.pdf
│   ├── Milestone2.pdf        # This milestone's report
│   └── Milestone3.pdf
│
├── requirements.txt          # pip dependencies
└── environment.yml           # Conda environment spec
