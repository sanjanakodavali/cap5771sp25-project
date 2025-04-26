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
├── Data/                     
│   ├── data_access_info.txt                # How to obtain full datasets
│   ├── drug200.csv                         # Sample interaction data
│   ├── medicine_dataset.csv.zip            # Raw data file
│   └── medicine_prescription_records.csv.zip
│
├── Scripts/                  
│   ├── Milestone1.py                       # Preprocessing & EDA
│   ├── Milestone2.py                       # Feature engineering & modeling (this milestone)
│   ├── Milestone3.py                       # Evaluation & interpretation
│   └── app.py                              # Streamlit dashboard
│
├── Report/                   
│   ├── Milestone1.pdf                      # Milestone1's report
│   ├── Milestone2.pdf                      # Milestone2's report
│   └── Milestone3.pdf                      # Milestone3's report
│
├── requirements.txt                        # pip dependencies
└── environment.yml                         # Conda environment spec

```

---

## 5. Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cap5771sp25-project.git
cd cap5771sp25-project

# 2. Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or using Conda:
conda env create -f environment.yml
conda activate cap5771
```

---

## 6. Running the Project

### 6.1 Preprocessing & EDA
```bash
python Scripts/Milestone1.py
```

### 6.2 Feature Engineering & Modeling
```bash
python Scripts/Milestone2.py
```

### 6.3 Advanced Evaluation
```bash
jupyter notebook Scripts/Milestone3.ipynb
```

### 6.4 Dashboard
```bash
streamlit run Scripts/app.py --server.port 8501
```
Open your browser at \`http://localhost:8501\`.

---

## 7. Demo Video
▶️ [YouTube Walkthrough (5 min)](https://youtu.be/REPLACE_WITH_YOUR_LINK)

---

## 8. Data Dependencies
- **Sample CSVs** in \`Data/\` (5% stratified sample)  
- **Full datasets** (>100 MB) available per \`Data/data_access_info.txt\`  

---

## 9. Collaboration & Submission
- **Repository name**: \`cap5771sp25-project\`  
- **Collaborators**: TA Jimmy (@JimmyRaoUF), Grader Daniyal (@abbasidaniyal), Dr. Cruz (@lcruz-cas), Dr. Grant (@cegme)  
- **Gradescope**: Ensured this exact URL is linked for grading  

---
