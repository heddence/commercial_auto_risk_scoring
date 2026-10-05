# Commercial auto claim risk

This project estimates the probability of at least one claim during a one-year commercial auto policy period. It covers Auto Physical Damage (APD) and Auto Liability (AL), using policy, fleet and prior-claims information.

## Approach

The notebook explores the data, removes duplicate records and invalid policy periods, excludes information that reveals the outcome, and prepares missing values and categorical inputs within each training pipeline. Valid policies active at data capture remain included.

Logistic regression, gradient boosting and a Poisson-based alternative were compared using earlier historical cohorts for model selection. Regularized logistic regression with a small set of derived features was selected, then evaluated on records captured in 2022. Finally, it was fitted on all 14,990 usable historical records to score 3,000 policies.

## Results

- **ROC AUC: 0.675**, compared with **0.574** for a benchmark assigning the historical claim rate for each coverage.
- **Log loss: 0.360**, a **3.9% improvement** over that benchmark. Lower log loss means better probability predictions.
- The highest-risk 10% of evaluated policy lines had a **29.2% claim occurrence rate**, about **2.3 times** the overall rate.

## Run

You can run this project in Databricks in Free Edition. To run locally, use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

With a Jupyter environment available, open `risk_scorer.ipynb` from the project directory and run the cells in order. Keep `risk_scoring.py` and the `data/` folder alongside it. The notebook generates the model, evaluation reports and `predictions.csv`.

## Limitations and next steps

The scoring data have no outcomes, so their predictive performance is unknown. Production work should run a shadow pilot and introduce monitoring. The notebook includes implemented improvements and a production roadmap.

ChatGPT/Codex assisted with review, documentation and presentation preparation. Reported results come from executing the Python workflow.
