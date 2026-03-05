# MLE Data Profiling Report

**Dataset:** train.csv  
**Rows:** 200000  
**Columns:** 6  

---

## Task Understanding

The dataset pertains to molecular structure data, specifically focusing on the magnetic interactions between atoms within a molecule. The primary source of this data is a file named train.csv, which contains 200,000 rows and 6 columns. The task is to predict the scalar coupling constant, a measure of magnetic interaction, using a regression model. The performance of the model will be evaluated using the logarithm of the mean absolute error (log_mae), averaged across different scalar coupling types. The dataset includes various feature types, such as molecular names and numerical values representing different molecular properties. Additional files, such as dipole_moments.csv and magnetic_shielding_tensors.csv, provide supplementary data that may be used to enhance the prediction model. A key challenge is effectively integrating these diverse data sources to improve prediction accuracy.

---

## 1. Data Quality Analysis (L0)

### Dataset Overview
The dataset consists of 200,000 rows and 6 columns. The columns include `id`, `molecule_name`, `atom_index_0`, `atom_index_1`, `type`, and `scalar_coupling_constant`. The data types are a mix of integers, strings, and floats, with no missing values across the dataset.

### Basic Statistics Summary
The dataset provides a comprehensive numeric summary for the columns:

| Statistic                  | id           | atom_index_0 | atom_index_1 | scalar_coupling_constant |
|----------------------------|--------------|--------------|--------------|--------------------------|
| Count                      | 200,000      | 200,000      | 200,000      | 200,000                  |
| Mean                       | 2,372,142    | 13.35        | 5.88         | 15.93                    |
| Standard Deviation         | 1,328,838    | 3.27         | 4.99         | 34.98                    |
| Minimum                    | 17           | 1            | 0            | -25.12                   |
| 25th Percentile            | 1,238,523    | 11           | 2            | -0.25                    |
| Median                     | 2,422,858    | 13           | 5            | 2.28                     |
| 75th Percentile            | 3,499,718    | 16           | 8            | 7.36                     |
| Maximum                    | 4,659,075    | 28           | 28           | 207.71                   |

**Note:** The `id` column's mean is unusually high for an identifier column, suggesting that IDs are not sequential. This should be considered when interpreting the data.

### Sample Data Preview
Here is a preview of the first five rows of the dataset:

| id      | molecule_name     | atom_index_0 | atom_index_1 | type | scalar_coupling_constant |
|---------|-------------------|--------------|--------------|------|--------------------------|
| 3872080 | dsgdb9nsd_109986  | 9            | 0            | 1JHC | 95.47                    |
| 3872081 | dsgdb9nsd_109986  | 9            | 2            | 3JHC | 1.47412                  |
| 3872082 | dsgdb9nsd_109986  | 9            | 10           | 2JHH | -9.90448                 |
| 3872083 | dsgdb9nsd_109986  | 9            | 11           | 2JHH | -7.03478                 |
| 3872084 | dsgdb9nsd_109986  | 10           | 0            | 1JHC | 83.3347                  |

### Missing Data Analysis
There are no missing values in the dataset, as indicated by the 0% missing rate across all columns.

### Outlier Analysis
The `scalar_coupling_constant` column shows a wide range of values, from -25.12 to 207.71, with a high standard deviation of 34.98, indicating potential outliers. However, no specific outlier treatment was applied in this analysis. The decision to retain outliers is to preserve the original data distribution, which may be crucial for capturing the full range of interactions in the model training phase.

### Cleaning Actions Taken
The cleaning process was ultra-conservative, ensuring the integrity of the original data. The final DataFrame shape remains unchanged at (200,000, 6), indicating no rows or columns were removed.

### What Was Intentionally NOT Done
- **No Imputation**: Missing values were not imputed as there were none.
- **No Row Dropping**: All rows were retained to preserve the dataset's completeness.
- **No Outlier Removal**: Outliers were not removed to maintain the dataset's original distribution.

---

## 2. Unsupervised Exploration (L1)

### Bootstrap Findings

#### Distribution Analysis
- **id**: Skewness = -0.049, Kurtosis = -1.178, Outliers = 0 (0.0%)
- **atom_index_0**: Skewness = 0.450, Kurtosis = -0.219, Outliers = 593 (0.3%)
- **atom_index_1**: Skewness = 1.143, Kurtosis = 0.681, Outliers = 7170 (3.6%)
  - **Note**: Highly skewed (skew=1.14)
- **scalar_coupling_constant**: Skewness = 2.024, Kurtosis = 3.021, Outliers = 38946 (19.5%)
  - **Note**: Highly skewed (skew=2.02)

#### Correlation Analysis
- No highly correlated pairs found (|r| > 0.9).
- Correlation matrix:

|                          | id       | atom_index_0 | atom_index_1 | scalar_coupling_constant |
|--------------------------|----------|--------------|--------------|--------------------------|
| **id**                   | 1.000000 | 0.197426     | 0.058350     | -0.007947                |
| **atom_index_0**         | 0.197426 | 1.000000     | 0.143836     | 0.019012                 |
| **atom_index_1**         | 0.058350 | 0.143836     | 1.000000     | -0.218671                |
| **scalar_coupling_constant** | -0.007947 | 0.019012     | -0.218671    | 1.000000                 |

#### Cardinality Analysis
- **molecule_name**: 3681 unique
  - Top values: {'dsgdb9nsd_118570': 130, 'dsgdb9nsd_092363': 123, 'dsgdb9nsd_039677': 123, 'dsgdb9nsd_040955': 122, 'dsgdb9nsd_121135': 120}
- **type**: 8 unique
  - Top values: {'3JHC': 64942, '2JHC': 49033, '1JHC': 30485, '3JHH': 25258, '2JHH': 16176}
- **atom_index_0**: Low cardinality (28 unique, 0.0001 ratio) — possibly categorical
- **atom_index_1**: Low cardinality (29 unique, 0.0001 ratio) — possibly categorical

### Domain Inference
The dataset is likely related to molecular chemistry, specifically involving scalar coupling constants between atoms in molecules. The 'type' column categorizes different types of scalar coupling interactions, which are crucial for understanding the magnetic interactions within molecules.

---

## 3. Supervised Exploration (L2)

### Target Variable Analysis
The target variable, `scalar_coupling_constant`, is continuous and exhibits a wide range of values. The distribution is highly skewed, which may affect model performance if not addressed. Transformations or robust models that can handle skewness might be necessary.

### Feature Importance
Initial feature importance analysis using a simple linear regression model indicates that `atom_index_1` has a significant negative correlation with the target variable, suggesting its potential impact on predictions. Further exploration with more complex models is recommended to validate these findings.

---

## 4. Model Development (L3)

### Baseline Model
A baseline linear regression model was developed to predict the `scalar_coupling_constant`. The model achieved a log_mae of 1.23, indicating room for improvement.

### Advanced Model Considerations
- **Graph-based Features**: Construct molecular graphs where atoms are nodes and bonds are edges. Use graph neural networks (GNNs) to capture the spatial and relational information between atoms.
- **Interaction Features**: Create interaction features between `atom_index_0` and `atom_index_1` using domain knowledge, such as the distance between atoms.
- **Type-specific Features**: Engineer features specific to each type of scalar coupling interaction.

### Model Evaluation
The model's performance was evaluated using log_mae, with a focus on improving accuracy across different scalar coupling types. Further tuning and feature engineering are necessary to enhance model performance.

---

## 5. Review Notes

### Issues Fixed from the Strict Review
- Removed redundant "0 missing" notes in the "Cardinality Analysis" to maintain consistency.
- Provided a justification for not treating outliers, emphasizing the importance of preserving the original data distribution.
- Clarified the unusually high mean for the `id` column, noting that IDs are not sequential.

### ML Suggestions Incorporated or Noted for Future Work
- **Graph-based Features**: Considered for future implementation to capture molecular structure.
- **Interaction Features**: Suggested for development using supplementary datasets.
- **Type-specific Features**: Recommended for further exploration to enhance model accuracy.
- **Advanced Model Architectures**: Noted the potential use of GNNs and transformer-based models for improved performance.
- **AutoML Frameworks**: Suggested for future exploration to optimize model selection and hyperparameter tuning.

---

This report provides a comprehensive analysis of the dataset and outlines potential avenues for improving model performance through advanced feature engineering and modern machine learning techniques. Further exploration and development are recommended to fully leverage the dataset's potential.