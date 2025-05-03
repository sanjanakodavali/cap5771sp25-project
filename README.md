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

## Setup Instructions

This project helps recommend the right medicine using a smart machine learning model. To see how it works, we’ve made a simple dashboard you can run right from your computer (or Google Colab). Just follow the steps below.

---

###  What You Need First (Before We Begin)

Make sure you have these ready:

- **Python** (version 3.7 or higher)
- **pip** (Python’s package installer)
- **Git** (optional — helps you download the project quickly)

If you're using **Google Colab**, don’t worry — you don’t need to install Python. We’ll give you a simple way to run it there too!

---

###  Step 1: Get the Project Files

#### Option 1: Using Git (Fastest)

```bash
git clone https://github.com/sanjanakodavali/cap5771sp25-project.git
```

#### Option 2: Manual Download

1. Go to the GitHub page for this project  
2. Click the green **"Code"** button  
3. Choose **Download ZIP**  
4. Unzip it and open the folder in VS Code or any editor

---

###  Step 2: Install the Tools (Dependencies)

#### 🖥️ If You're on a Computer (Local Setup)

1. Open a terminal in the project folder  
2. Run this to create a private workspace:

```bash
python -m venv venv
```

3. Activate the environment:

- **Windows:**
```bash
venv\Scripts\activate
```
- **Mac/Linux:**
```bash
source venv/bin/activate
```

4. Install all necessary tools:

```bash
pip install -e .
```

---

####  If You're Using Google Colab

Just copy and run this in a code cell:

```python
!pip install streamlit shap scikit-learn xgboost matplotlib plotly joblib
```

Note: You can't launch the full dashboard inside Colab, but you can test the models and SHAP visualizations.

---

### ▶️Step 3: Run the Dashboard

If you're on your computer, run this to open the dashboard in your browser:

```bash
streamlit run dashboard/app.py
```

You'll see:

-  Model scores like Accuracy and F1
-  ROC and PR curves
-  Top features influencing predictions
-  SHAP plots to explain decisions

---

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
python Scripts/Milestone3.py
```

### 6.4 Dashboard
```bash
streamlit run Scripts/app.py --server.port 8501
```
Open your browser at \`http://localhost:8501\`.

---

## 7. Demo Video
https://drive.google.com/file/d/1uEv8VSEn2mmEMWSAeHaylA3wwlR9mKbP/view?usp=sharing
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
