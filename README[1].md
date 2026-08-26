# Student Performance Analytics

Complete source code for the Student Performance Analytics project.

## Main files

- `Student_Performance_Analytics.ipynb` — complete Jupyter Notebook
- `student_performance_analytics.py` — Python source code
- `requirements.txt` — required Python packages

## Dataset

UCI Student Performance Dataset:
https://archive.ics.uci.edu/dataset/320/student

The project uses the Portuguese-language student performance dataset with 649 records.

## Method

- Data cleaning
- Exploratory Data Analysis
- Feature preparation
- Performance categorization
- 80/20 train-test split
- Random Forest Classification
- Accuracy, precision, recall, F1-score and confusion matrix
- Feature importance
- Dashboard-ready analytics

## Performance categories

- Low: final grade 0-9
- Average: final grade 10-14
- High: final grade 15-20

## Run

Install dependencies:

`pip install -r requirements.txt`

Open the notebook:

`jupyter notebook Student_Performance_Analytics.ipynb`

Or run:

`python student_performance_analytics.py`
