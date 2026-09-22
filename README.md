# Student Performance Prediction

A machine learning project that predicts a student's academic **Performance Index** from study habits and lifestyle factors, using multiple linear regression.

Built as part of the **IBM SkillsBuild Data Analytics with AI Academic Internship Program**, conducted by BharatCares in association with AICTE.

## Project Description

Student performance is shaped by more than just study time — prior achievement, rest, and practice all play a role. This project trains a multiple linear regression model on 10,000 student records to estimate a student's Performance Index (0–100) from five inputs: hours studied, previous test scores, extracurricular participation, sleep hours, and practice papers solved.

The trained model is also deployed as a small standalone web app (`predictor.html`) so new predictions can be made interactively in a browser, without running any Python code.

## Dataset

**Source:** [Student Performance (Multiple Linear Regression) — Kaggle](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data)

10,000 student records with the following columns:

| Column | Description |
|---|---|
| Hours Studied | Average daily study time |
| Previous Scores | Score on the most recent exam (0–100) |
| Extracurricular Activities | Whether the student participates (Yes/No) |
| Sleep Hours | Typical nightly sleep duration |
| Sample Question Papers Practiced | Number of practice papers solved |
| Performance Index | Target variable — overall performance score (0–100) |

## Technologies Used

- **Python 3** — core language
- **pandas / numpy** — data loading and manipulation
- **scikit-learn** — model training (`LinearRegression`), train/test split, evaluation metrics
- **matplotlib / seaborn** — exploratory data analysis and visualizations
- **HTML / CSS / JavaScript** — standalone interactive predictor page

## Setup & Run Instructions

### 1. Clone or download the project folder

- `Shaily_StudentPerformancePrediction.py` 
- `Student_Performance.csv`
- `requirements.txt`

### 2. Create a virtual environment 

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python Shaily_StudentPerformancePrediction.py
```

Or open the `.ipynb` file in Jupyter Notebook / VS Code and run all cells.

This will:
- Load and clean the dataset
- Train a linear regression model on an 80/20 train-test split
- Print evaluation metrics (MAE, MSE, RMSE, R²)
- Generate and save three plots: a correlation heatmap, an actual-vs-predicted scatter plot, and a hours-studied-vs-performance scatter plot
- Predict the Performance Index for a sample new student

### 5. Use the interactive predictor

Open `predictor.html` directly in any web browser — no server or install needed. Enter a student's details and get an instant predicted Performance Index, computed client-side using the trained model's coefficients.

## Key Results

| Metric | Value |
|---|---|
| MAE | 1.61 |
| MSE | 4.08 |
| RMSE | 2.02 |
| R² Score | 0.989 |

The model explains about 98.9% of the variance in student performance, with Previous Scores and Hours Studied as the strongest predictors.

## Author

Shaily — IBM SkillsBuild Data Analytics with AI Academic Internship Program (BharatCares × AICTE)
