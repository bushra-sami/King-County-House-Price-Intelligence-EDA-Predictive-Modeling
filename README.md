# King County House Price Intelligence: EDA & Predictive Modeling

> **End-to-end residential real estate analysis** — from raw sales data to a production-ready price prediction model — covering King County, Washington (2014). Combines statistical hypothesis testing, domain-driven data cleaning, interpretable feature engineering, and dual-model comparison to explain and predict property values with ~80% accuracy.

---

##  Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Data Cleaning & Validation](#data-cleaning--validation)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Model Results](#model-results)
- [Key Business Insights](#key-business-insights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

This project performs a complete data science lifecycle on King County residential home sales data, addressing the fundamental real estate question:

> *What drives house prices in the Seattle metropolitan area — and how accurately can we predict them?*

The workflow progresses through four stages:

1. **Data Integrity** — Systematic cleaning using both IQR statistics and hand-crafted domain rules
2. **Analytical EDA** — Seven structured business questions answered with statistical tests and rich visualizations
3. **Predictive Modeling** — Linear Regression (interpretable baseline) vs. Random Forest (performance-optimized)
4. **Business Reporting** — Plain-language findings suitable for non-technical stakeholders

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | King County, WA residential property sales |
| **Period** | May – July 2014 *(single-year snapshot)* |
| **Raw Records** | 4,600 rows |
| **Clean Records** | ~4,450 (after domain + IQR filtering) |
| **Features** | 18 columns |
| **Target Variable** | `price` (USD) |
| **Geography** | Seattle metro area, WA, USA |

### Feature Dictionary

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Sale date |
| `price` | float | Sale price in USD — **target variable** |
| `bedrooms` | float | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms (0.5 increments) |
| `sqft_living` | int | Interior living area (sq ft) |
| `sqft_lot` | int | Total lot/land area (sq ft) |
| `floors` | float | Number of floors |
| `waterfront` | int | Waterfront property flag (0/1) |
| `view` | int | View quality grade (0–4) |
| `condition` | int | Property condition rating (1–5) |
| `sqft_above` | int | Above-ground living area (sq ft) |
| `sqft_basement` | int | Basement area (sq ft) |
| `yr_built` | int | Year of original construction |
| `yr_renovated` | int | Year of last renovation (0 = never) |
| `street` | str | Street address |
| `city` | str | City name (Seattle, Bellevue, Renton, …) |
| `statezip` | str | State + ZIP code |
| `country` | str | Country (all USA — dropped as constant) |

**Price range:** $0 – $26.6M | **Mean:** ~$552K | **Median:** ~$461K

---

## Project Architecture

```
Raw CSV (4,600 rows, 18 features)
         │
         ▼
┌─────────────────────────┐
│   Data Cleaning Layer   │
│  • Drop constant cols   │
│  • Remove zero prices   │
│  • Domain-rule filter   │
│  • IQR outlier removal  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Feature Engineering    │
│  • Log transforms       │
│  • Age calculation      │
│  • Renovation flag      │
│  • total_sqft           │
│  • luxury_score         │
│  • price_per_sqft       │
│  • sale_month           │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Exploratory Analysis   │
│  7 Business Questions   │
│  + Statistical Tests    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   ML Pipeline (80/20)   │
│  • OHE + Target Enc.    │
│  • StandardScaler       │
│  • LinearRegression     │
│  • RandomForestRegressor│
└──────────┬──────────────┘
           │
           ▼
   Evaluation & Report
```

---

## Data Cleaning & Validation

Cleaning followed a two-phase approach — statistical detection followed by domain expert rules — ensuring no valid data was lost while removing records that would corrupt model training.

### Phase 1 — Structural Cleanup

- **Constant feature removal:** `country` (all `USA`) and `waterfront` were evaluated for cardinality. `country` was dropped as it provides zero signal.
- **Zero-price removal:** 49 records with `price = 0` were removed — logically invalid in a market context.

### Phase 2 — Domain-Driven Outlier Filtering

A composite `domain_flag` was constructed using real estate domain knowledge:

```python
domain_flag = (
    (price < 35_000)               |   # sub-market threshold
    (sqft_living < 300)            |   # physically implausible dwelling
    (bedrooms > 12)                |   # extreme misreport / multi-family
    (waterfront == 1 & price < 650_000)  |   # pricing contradiction
    (sqft_living > 8_000 & price < 1_800_000) |  # size-price mismatch
    (yr_renovated > 2015)          |   # future date — data error
    (yr_built > 2015)                  # future date — data error
)
```

### Phase 3 — Statistical IQR Clipping

Standard IQR-based bounds were applied to `price` after domain filtering to remove residual statistical outliers without domain justification.

**Cleaning Impact Summary:**

| Stage | Records Retained |
|-------|-----------------|
| Raw data | 4,600 |
| After zero-price removal | ~4,551 |
| After domain filtering | ~4,450 |
| After IQR clipping | ~4,350 (estimated) |

>  No null values were present in the original dataset — confirmed via `.isnull().sum()`.

---

## Feature Engineering

Eight new features were constructed from the raw attributes:

| New Feature | Formula | Rationale |
|-------------|---------|-----------|
| `log_price` | `log1p(price)` | Normalizes right-skewed price distribution; used as model target |
| `log_sqft_living` | `log1p(sqft_living)` | Compresses scale for correlated size features |
| `log_sqft_lot` | `log1p(sqft_lot)` | Same as above for lot area |
| `age` | `2014 − yr_built` | Converts year into an interpretable age metric |
| `renovated_flag` | `(yr_renovated > 0).astype(int)` | Binary: was the property ever renovated? |
| `total_sqft` | `sqft_above + sqft_basement` | Unified indoor square footage measure |
| `luxury_score` | `waterfront + view + (bathrooms / (bedrooms + 0.001))` | Composite luxury indicator |
| `price_per_sqft` | `price / sqft_living` | Density-normalized price for EDA and diagnostics |
| `sale_month` | `date.dt.month` | Captures seasonality in buyer demand |

---

## Exploratory Data Analysis

Seven business questions were answered with statistical rigor:

### Q1 — Living Area vs. Price
- **Pearson correlation:** ~0.70 between `sqft_living` and `price`
- Living area explains approximately 66–68% of price variance
- Price-per-sqft was analyzed across 10 equal-frequency (decile) bins
- *Visual:* Scatter plot with linear regression overlay

### Q2 — Waterfront Premium
- Waterfront homes sell for **2.5× to 4× more** than non-waterfront equivalents
- Statistical separation is large and consistent across the entire price distribution
- *Visual:* Violin + strip plot with log-scale y-axis

### Q3 — Bedrooms & Bathrooms Interaction
- Joint heatmap of average price by bedroom count (index) × bathroom bins (columns)
- Reveals that **bathroom count has a stronger marginal effect than bedroom count alone**
- *Visual:* Annotated heatmap (YlGnBu palette)

### Q4 — Renovation & Age Effect
- Renovated properties command a measurable premium over non-renovated equivalents
- Age-bin analysis shows an inverse U-curve: mid-century homes (1940s–1970s) compete well with newer construction in certain sub-markets
- *Visual:* Boxen plot of log-price by age quartile

### Q5 — Geographic Pricing
- **Most expensive cities:** Medina, Clyde Hill, Mercer Island, Bellevue
- **Most affordable cities:** Auburn, Kent, Federal Way
- Seattle dominates transaction volume (1,573 of 4,600 sales)
- Price-per-sqft ranking largely follows mean price ranking
- *Visual:* Horizontal bar chart (top 15 / bottom 15 cities)

### Q6 — View Grade Premium
- View grade shows a strong monotonic relationship with price (grade 0 → grade 4)
- **ANOVA confirms statistical significance** across all view grades (*p* < 0.001)
- Effect persists within each living-area quartile (confirmed with stratified ANOVA)
- *Visual:* Point plot with 95% CI; stratified heatmap

### Q7 — Seasonal Pricing Patterns
- **Spring/Summer (April–July):** Higher sale prices and higher transaction volume
- **Winter (December–February):** Consistently lower prices
- Both ANOVA and Kruskal-Wallis tests confirm seasonal variation is statistically significant (*p* < 0.05)
- *Visual:* Dual-line chart (mean & median price by month) + monthly box plot

---

## Machine Learning Pipeline

### Feature Set (12 input features)

```
sqft_living, bedrooms, bathrooms, floors,
waterfront, view, condition, age,
renovated_flag, total_sqft, city, sale_month
```

### Train/Test Split

- **80% training / 20% test** with `shuffle=True`, `random_state=42`
- No stratification — regression target; shuffling ensures independence

### Encoding

| Feature Type | Strategy |
|-------------|---------|
| Low-cardinality ordinals (`waterfront`, `view`, `condition`, `renovated_flag`, `sale_month`) | One-Hot Encoding (`drop_first=True`) |
| High-cardinality nominal (`city`) | Target Encoding (mean `log_price` per city, computed on train set only) |

Column alignment between train and test sets performed via `.reindex()` to handle rare categories absent in the test split.

### Scaling

`StandardScaler` (zero mean, unit variance) applied to all continuous numeric features:
`sqft_living`, `bedrooms`, `bathrooms`, `floors`, `age`, `total_sqft`, `city_encoded`

Scaler fitted **on training data only** — transform applied to test data to prevent leakage.

### Models

**Model 1 — Linear Regression** *(interpretable baseline)*
```python
LinearRegression()
```

**Model 2 — Random Forest Regressor** *(primary model)*
```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

---

## Model Results

All metrics evaluated on the **20% held-out test set**.

| Metric | Linear Regression | Random Forest |
|--------|:-----------------:|:-------------:|
| R² (log scale) | **0.6853** | 0.6652 |
| RMSE (log scale) | — | — |
| MAE (original $) | $90,316 | **$86,886** |
| MAPE | — | **19.36%** |
| Approx. Accuracy | — | **~80.6%** |

### Model Selection Rationale

> **Random Forest is the recommended production model.** Although Linear Regression achieves a slightly higher R², Random Forest delivers a lower dollar-denominated MAE ($86,886 vs. $90,316), which is the metric that matters in real estate valuation. Its ability to capture non-linear interactions (e.g., view grade × living area) provides more reliable predictions across the price distribution.

### Top Feature Importances (Random Forest)

The model's most influential features for price prediction were:

1. `city_encoded` — geographic location encodes neighborhood desirability
2. `sqft_living` — total living area
3. `total_sqft` — above-ground + basement area
4. `age` — property age at time of sale
5. `bathrooms` — bathroom count
6. `view` — scenic view quality grade
7. `condition` — property maintenance rating
8. `sale_month` — seasonality of purchase

*(Rankings based on `feature_importances_` from the fitted Random Forest model.)*

---

## Key Business Insights

| Insight | Quantified Impact |
|---------|------------------|
| 🏗️ **Size drives value** | Every 100 sqft increase in living area adds measurable price premium (r ≈ 0.70) |
| 🌊 **Waterfront is the strongest premium** | Waterfront homes price at **2.5–4× non-waterfront** equivalents |
| 👁️ **Views command a clear premium** | Grade-4 view homes significantly outprice Grade-0 (ANOVA p < 0.001) |
| 🔨 **Renovation pays off** | Renovated properties consistently sell above non-renovated comparables |
| 📍 **Location dominates all else** | City alone is the single most important predictive feature |
| 📅 **Spring is the best time to sell** | April–July shows peak prices; December–February is weakest |
| 🧮 **Model accuracy** | Random Forest achieves ~80.6% approximate accuracy (100 − MAPE) with MAE ≈ $87K |

---

## Tech Stack

| Category | Library / Tool |
|----------|---------------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Statistical testing | `scipy.stats` (ANOVA, Kruskal-Wallis) |
| Machine learning | `scikit-learn` |
| Models | `LinearRegression`, `RandomForestRegressor` |
| Preprocessing | `StandardScaler`, `train_test_split` |
| Metrics | `r2_score`, `mean_squared_error`, `mean_absolute_error` |
| Environment | Jupyter Notebook (Google Colab) |
| Language | Python 3.x |

---

## Project Structure

```
king-county-house-price/
│
├── data/
│   ├── data (1).csv           # Raw input dataset (4,600 records)
│   └── cleaned_data.csv       # Exported clean dataset (generated by notebook)
│
├── notebooks/
│   └── data_analysis_project.ipynb   # Main analysis & modeling notebook
│
└── README.md                  # This file
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Running the Notebook

```bash
# Clone the repository
git clone https://github.com/<your-username>/king-county-house-price.git
cd king-county-house-price

# Launch Jupyter
jupyter notebook notebooks/data_analysis_project.ipynb
```

**Or open directly in Google Colab:**
> Update the CSV path in the `Data Loading` cell from `/content/data (1).csv` to your local or Drive path.

### Expected Outputs

After running all cells, you will have:
- 15+ annotated visualizations saved inline in the notebook
- Console output for all statistical tests (ANOVA / Kruskal-Wallis)
- Trained Linear Regression and Random Forest models with printed metrics
- `cleaned_data.csv` exported to the working directory

---

## Limitations & Future Work

### Current Limitations

| Limitation | Description |
|-----------|-------------|
| **Single-year snapshot** | All 4,600 sales are from 2014 only — temporal generalization is limited |
| **No spatial features** | Latitude/longitude coordinates not utilized (available in extended King County dataset) |
| **No hyperparameter tuning** | Random Forest uses fixed hyperparameters; GridSearchCV or Optuna could improve performance |
| **Target encoding leakage risk** | City means are computed on the full training set — cross-validated target encoding would be more rigorous |
| **No cross-validation** | Single hold-out evaluation; k-fold CV would yield more stable performance estimates |

### Suggested Improvements

- **Add geospatial features** — distance to downtown Seattle, school district quality scores, walkability index
- **Extend temporal coverage** — multi-year data would enable time-series trend modeling
- **Gradient Boosting** — XGBoost or LightGBM typically outperform Random Forest on tabular real estate data
- **SHAP explanations** — add model interpretability layer for individual property valuations
- **Automated pipeline** — wrap preprocessing + model in a `scikit-learn Pipeline` object for deployment readiness
- **Interactive dashboard** — Streamlit or Gradio app for real-time price estimation

---

## Author

*Analysis and modeling developed as a data science portfolio project.*

---

## License

This project is licensed under the MIT License.

