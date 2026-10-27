# Fake Job & Internship Detection Using Machine Learning

A Predictive Analytics college project that classifies job postings as **genuine (0)** or **potentially fraudulent (1)** using four machine learning models and compares their performance.

## Dataset

- Name: Real or Fake? Fake Job Posting Prediction (EMSCAD)
- Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Size: 17,880 postings, 18 columns
- Target column: `fraudulent` (0 = genuine, 1 = fraudulent)
- Class imbalance: 17,014 genuine (95.16%) and 866 fraudulent (4.84%)

The dataset belongs to its original authors and is included here only for academic use.

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (LinearSVC)

All four models use the same stratified train-test split (80/20, `random_state=42`) and the same preprocessing pipeline.

## Methodology

1. **Data understanding:** shape, data types, missing values, duplicates, class distribution.
2. **EDA:** class balance, missing values, employment type, remote work, company logo and screening questions vs fraud rate.
3. **Preprocessing:**
   - Removed 281 exact duplicate postings (17,880 to 17,599 rows) to avoid train-test leakage.
   - Combined `title`, `company_profile`, `description`, `requirements` and `benefits` into one text column and applied TF-IDF (max 5000 features).
   - Applied OneHotEncoder to categorical columns (employment type, experience, education, industry, function, country).
   - Created `has_salary` (0/1) from `salary_range`; dropped `job_id` and `department`.
   - Used `ColumnTransformer` + `Pipeline`, fitted on training data only, so there is no data leakage.
4. **Training:** 14,079 training rows and 3,520 test rows. `class_weight="balanced"` is used to handle imbalance.
5. **Evaluation:** accuracy, precision, recall, F1-score and confusion matrix, measured on the fraudulent class.

## Results (Test Set: 3,520 postings, 171 fraudulent)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| SVM (LinearSVC) | 0.9801 | 0.7760 | 0.8304 | 0.8023 |
| Random Forest | 0.9790 | 1.0000 | 0.5673 | 0.7239 |
| Logistic Regression | 0.9622 | 0.5688 | 0.9181 | 0.7025 |
| Decision Tree | 0.9693 | 0.6760 | 0.7076 | 0.6914 |

Confusion matrix values:

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 3230 | 119 | 14 | 157 |
| Decision Tree | 3291 | 58 | 50 | 121 |
| Random Forest | 3349 | 0 | 74 | 97 |
| SVM (LinearSVC) | 3308 | 41 | 29 | 142 |

### Key Findings

- **Best F1-score:** SVM (0.8023), with the best balance between precision and recall.
- **Highest recall (catches the most fraud):** Logistic Regression (0.9181), but with many false alarms.
- **Random Forest** never wrongly flagged a genuine posting (precision 1.0) but missed many fraudulent ones (recall 0.5673).
- Accuracy alone is misleading here because 95% of postings are genuine.

## Limitations

- Results are based on a single train-test split.
- Only 171 fraudulent postings in the test set, so small changes can affect the metrics.
- No hyperparameter tuning was done.
- The dataset is old and only covers English postings.
- A "fraudulent" prediction is a warning signal, not proof. Always verify a job through the official company website or HR.

## Project Structure

```
fake-job-detection/
├── data/
│   └── fake_job_postings.csv
├── notebooks/
│   └── fake_job_detection.ipynb
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/mridul2172/fake-job-detection.git
   cd fake-job-detection
   ```
2. Install the libraries:
   ```
   pip install -r requirements.txt
   pip install jupyter ipykernel
   ```
3. Open `notebooks/fake_job_detection.ipynb` in Jupyter or VS Code and use **Run All**.

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook