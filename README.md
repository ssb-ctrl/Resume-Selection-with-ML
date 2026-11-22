# Resume Selection with ML

An NLP pipeline that classifies resumes as **flagged or not flagged** from their text, using bag-of-words features and Multinomial Naive Bayes.

## Dataset
- **125 resumes** with a text column and a label column
- **92 not flagged, 33 flagged** (about 26% flagged), no missing values

## Workflow
1. **Load and explore:** class counts and label encoding (flagged = 1, not flagged = 0)
2. **Clean text:** lowercase tokenization with Gensim, removal of NLTK and Gensim stop words plus custom ones (e.g. "email", "com"), and removal of tokens shorter than 3 characters
3. **Visualize:** count plot of classes and word clouds for each class
4. **Features:** bag-of-words with `CountVectorizer`
5. **Model:** `MultinomialNB(alpha=3)` on an 80/20 train/test split
6. **Evaluate:** confusion matrices, classification report, accuracy, precision, recall and F1

## Results
| Metric | Score |
|---|---|
| Train accuracy | 93% |
| Test accuracy | 88% |

Test-set classification report (25 resumes):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (not flagged) | 0.88 | 1.00 | 0.93 | 21 |
| 1 (flagged) | 1.00 | 0.25 | 0.40 | 4 |

The model catches few flagged resumes (recall 0.25). The dataset is small and imbalanced, and the test set has only 4 flagged resumes, so results are unstable. No random seed is fixed.

## Limitations and next steps
- Use a stratified split and cross-validation, given the tiny test set
- Report PR-AUC and ROC-AUC, and tune the decision threshold for recall
- Try TF-IDF features and compare against Logistic Regression
- Models trained on past labels can inherit their bias, so predictions should support human review, not replace it

## Project Structure
```
├── Resume_Selection.ipynb       # analysis and model
├── data/                        # dataset
├── LICENSE
└── README.md
```

## How to Run
```bash
git clone https://github.com/ssb-ctrl/Resume-Selection-with-ML.git
cd Resume-Selection-with-ML
pip install pandas numpy matplotlib seaborn nltk gensim scikit-learn wordcloud jupyter
jupyter notebook
```
