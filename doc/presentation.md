You are working inside an existing research repository containing CRSP Daily data, experiment configurations, model code, saved metrics, tables, and plots.

Your task is to inspect the repository, analyze the actual experimental results, and create a concise academic presentation suitable for presenting the project to a professor.

## Objective

Create a Beamer-style research presentation that clearly explains:

1. Data
2. Exact research problem
3. Methodology / toolkit
4. Experimental configurations and results
5. Analysis, limitations, and conclusions

The presentation should tell a coherent research story rather than simply list experiments.

---

## Step 1 — Audit the repository first

Before creating slides:

* locate the CRSP Daily dataset or processed dataset;
* identify the sample period;
* identify the prediction target;
* identify features used;
* identify train / validation / test / out-of-sample splits;
* locate experiment configuration files;
* locate saved metrics, logs, tables, figures, and model outputs;
* identify which experiments were successfully completed;
* identify missing, failed, or incomplete experiments.

Create a short internal summary of the experimental structure before generating slides.

Do NOT invent experimental results.

If an expected result is missing, explicitly mark it as:

`Experiment pending / result unavailable`

---

# Presentation Structure

## Part 1 — Data

Explain the CRSP Daily dataset used in the project.

Include, when available:

* sample period;
* number of stocks;
* number of observations;
* relevant variables;
* missing-value handling;
* universe filters;
* preprocessing;
* feature construction;
* target construction.

Clearly define the prediction target.

For regression:

$$
y_{i,t+1} = r_{i,t+1}
$$

For classification:

$$
y_{i,t+1}
=
\mathbf{1}\{r_{i,t+1}>0\}.
$$

Explain exactly what information at time \(t\) is allowed to predict \(t+1\).

---

## Part 2 — Research Problem

State the research question precisely.

Main question:

> How well can next-trading-day stock returns or return directions be predicted from CRSP Daily information using statistical and machine-learning models?

Separate the project into two tasks:

### Regression

Predict next-day return:

$$
\hat r_{i,t+1}.
$$

### Classification

Predict next-day return direction:

$$
P(r_{i,t+1}>0).
$$

Explain why this is difficult:

* very low signal-to-noise ratio;
* cross-sectional and temporal dependence;
* nonstationarity;
* transaction-cost considerations;
* risk of data leakage and overfitting.

---

## Part 3 — Experimental Toolkit

Organize the toolkit into a systematic experiment matrix.

### Models

Include only models actually implemented or configured in the repository.

Possible examples:

* Linear regression
* Logistic regression
* regularized regression
* tree-based models
* recurrent neural networks
* Transformer-based models

### Objective functions

Identify which objectives were actually tested:

* MSE
* MAE
* Huber loss
* cross entropy
* Sharpe-related objective
* WorldQuant-style fitness objective

Do not imply an objective was tested unless results exist.

### Validation

Explain the chronological evaluation framework:

Train → Validation → Test → Out-of-Sample

If walk-forward validation is implemented, visualize it.

Explain why random train/test splits are inappropriate for this problem.

---

## Part 4 — Experimental Results

Organize results by experiment configuration rather than by code file.

Construct a master experiment table similar to:

| Experiment | Model | Task | Objective | Features | Period | Main Result |
| ---------- | ----- | ---- | --------- | -------- | ------ | ----------- |

Then analyze the experiments systematically.

For regression, report available metrics such as:

* MSE
* MAE
* Pearson IC
* Rank IC
* out-of-sample \(R^2\)

For classification, report available metrics such as:

* accuracy
* cross entropy
* AUC
* directional accuracy

For portfolio / economic evaluation, report when available:

* long-short return
* Sharpe ratio
* turnover
* drawdown
* WorldQuant-style fitness

Prefer plots over large tables.

Important plots may include:

1. IC through time
2. Rank IC through time
3. cumulative long-short portfolio return
4. model comparison
5. train vs validation vs test performance
6. prediction-vs-realized-return plots
7. experiment metric comparison
8. walk-forward out-of-sample performance

For every major figure, add one concise takeaway sentence.

Example:

> The model obtains positive training IC but near-zero out-of-sample IC, suggesting substantial overfitting.

Only make conclusions supported by actual results.

---

## Part 5 — Analysis

Compare experiments and answer questions such as:

* Which models generalize best out of sample?
* Does model complexity improve predictive performance?
* Are regression and classification results consistent?
* Which objective functions appear most stable?
* Is Rank IC more informative than raw prediction error?
* Does statistical prediction improvement translate into portfolio improvement?
* How stable are results across time?

Explicitly distinguish:

### Statistical significance

from

### Economic significance

and

### In-sample performance

from

### Out-of-sample performance.

---

## Methodological Audit

Include a slide checking for possible research errors:

* look-ahead bias;
* target leakage;
* incorrect lagging;
* survivorship bias;
* universe-selection bias;
* random rather than chronological splitting;
* normalization using future observations;
* hyperparameter tuning on the test set;
* transaction costs being ignored.

If any issue is detected in the repository, flag it rather than hiding it.

---

## Conclusion

The conclusion should contain approximately 3–5 evidence-based statements.

Structure them as:

### What worked

Which configurations produced meaningful results?

### What did not work

Which approaches failed to generalize?

### Main empirical finding

What does the experiment actually tell us about CRSP Daily return prediction?

### Limitations

What prevents stronger conclusions?

### Next experiment

What is the single most informative next experiment to run?

---

# Presentation Style

Target audience:

A professor familiar with statistics, machine learning, and quantitative finance.

Use an academic research-seminar style.

Requirements:

* approximately 12–18 slides;
* one main idea per slide;
* minimal prose;
* figures and tables preferred;
* mathematically precise notation;
* no marketing language;
* no exaggerated claims;
* clearly distinguish evidence from interpretation.

Suggested flow:

1. Title
2. Motivation
3. CRSP data
4. Prediction target
5. Experimental design
6. Chronological split
7. Model / objective matrix
   8–12. Main experimental results
8. Cross-experiment comparison
9. Robustness / methodological audit
10. Main findings
11. Limitations
12. Next experiments

---

# Deliverables

Create:

`presentation/main.tex`

with a compilable LaTeX Beamer presentation.

Also create:

`presentation/figures/`

for generated or copied figures.

Create:

`presentation/experiment_summary.md`

containing:

* all experiments discovered;
* configuration of each experiment;
* location of source result files;
* key metrics;
* which experiments appear in the slides;
* missing or incomplete experiments.

Compile the presentation if the environment supports LaTeX.

Do not modify the original experimental results.

The slides must be reproducible from the existing repository.

Most importantly:

**Do not invent data, metrics, experiments, or conclusions. Every quantitative claim in the presentation must be traceable to an actual experiment output in the repository.**
