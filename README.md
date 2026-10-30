# Real vs Fake Job Post Prediction

Machine learning project that classifies online job postings as real or fake using NLP (TF-IDF) with Decision Tree and Random Forest models.

## Problem
Fake job ads trick job seekers into paying money or sharing bank details. Checking them by hand is slow and unreliable. This project builds models that flag suspicious postings automatically.

## Dataset
- Kaggle "Real / Fake Job Posting Prediction" (EMSCAD), 17,880 job posts: 17,014 real and 866 fake (about 4.8% fake)
- Text fields: title, location, department, salary range, company profile, description, requirements, benefits, employment type, experience, education, industry, function
- Target: `fraudulent` (0 = real, 1 = fake)
- Source: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

## Approach
1. Combined all text fields into one content column
2. Cleaned text: expanded contractions, removed HTML and special characters, removed stop words, lowercased
3. Split 70% train / 30% test (stratified) BEFORE any balancing, so the test set stays untouched
4. Converted text to TF-IDF features (20,000 features, unigrams and bigrams), fitted on training data only
5. Handled class imbalance with `class_weight='balanced'`
6. Trained Decision Tree and Random Forest (200 trees)
7. Evaluated on the test set (5,364 posts, 260 fake)

## Results
Precision, recall and F1 are for the fake class. Accuracy alone is misleading here because 95% of posts are real.

| Model         | Accuracy | Precision | Recall | F1   | ROC-AUC |
|---------------|----------|-----------|--------|------|---------|
| Decision Tree | 95.67%   | 54.4%     | 66.9%  | 0.60 | 0.820   |
| Random Forest | 98.06%   | 100%      | 60.0%  | 0.75 | 0.993   |

Confusion matrix (test set):

| Model         | Real correct | Real flagged as fake | Fake missed | Fake caught |
|---------------|--------------|----------------------|-------------|-------------|
| Decision Tree | 4,958        | 146                  | 86          | 174         |
| Random Forest | 5,104        | 0                    | 104         | 156         |

Key findings:
- Random Forest is the better model overall: higher accuracy, F1 and AUC, and it never flagged a real job as fake.
- Decision Tree catches slightly more fake posts (174 vs 156) but wrongly flags 146 real jobs.
- Random Forest is cautious: it misses 40% of fake posts. Lowering its decision threshold would catch more fakes at the cost of some false alarms.

Charts are in the `screenshots` folder: class distribution, confusion matrices, ROC curve, top features.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn, Google Colab

## How to Run
1. Download `fake_job_postings.csv` from the Kaggle link above
2. Open `fake_job_prediction.ipynb` in Google Colab
3. Run all cells and upload the CSV when asked

## Future Work
- Tune the Random Forest threshold to improve recall
- Try Gradient Boosting and Logistic Regression
- Use word embeddings or BERT for better text features
- Deploy as a small web app

## Author
Subhan | https://linkedin.com/in/subhan-basha-proddutur-36686725a
