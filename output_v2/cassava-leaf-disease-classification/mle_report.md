# MLE Data Profiling Report

**Dataset:** train.csv  
**Rows:** 18721  
**Columns:** 2  

---

## Task Understanding

The dataset consists of images of cassava leaves collected from farmers in Uganda, aimed at classifying the leaves into one of four disease categories or identifying them as healthy. The task is a classification problem with the target variable being the 'label' column, and the performance metric is accuracy. The dataset includes 18,721 rows and 2 columns in the train.csv file, with image files primarily in .jpg format. A key challenge is handling the mixed data modalities, including .jpg and .tfrec files, and ensuring accurate classification to assist farmers in preventing crop damage.

---

## 1. Data Quality Analysis (L0)

### Dataset Overview

The dataset consists of a total of 21,417 files with a combined size of 6.18 GB. The files are distributed across various formats, including 21,397 image files (.jpg), 16 TFRecord files (.tfrec), 2 CSV files (.csv), 1 JSON file (.json), and 1 Markdown file (.md). The directory structure includes separate folders for test and train images, as well as TFRecord files.

The tabular metadata is contained in a CSV file with a shape of 18,721 rows and 2 columns: `image_id` (string) and `label` (integer).

### Basic Statistics Summary

The `label` column contains integer values with the following statistics:

| Statistic | Value     |
|-----------|-----------|
| Count     | 18,721    |
| Mean      | 2.655841  |
| Std Dev   | 0.985894  |
| Min       | 0         |
| 25%       | 2         |
| 50%       | 3         |
| 75%       | 3         |
| Max       | 4         |

### Sample Data Preview

Below is a preview of the first five rows of the tabular data:

| image_id       | label |
|----------------|-------|
| 1000015157.jpg | 0     |
| 1000201771.jpg | 3     |
| 100042118.jpg  | 1     |
| 1000723321.jpg | 1     |
| 1000812911.jpg | 3     |

### Missing Data Analysis

The dataset does not contain any missing values in the `image_id` or `label` columns. However, there are 73 entries in the CSV file that do not have corresponding image files, indicating potential data integrity issues.

### Outlier Analysis

Outlier detection using the Interquartile Range (IQR) method identified 939 entries (5.0%) in the `label` column that contribute to a distribution issue, such as class imbalance, rather than true statistical outliers. All values are within the expected range of [0, 4].

### Cleaning Actions Taken

The cleaning process was ultra-conservative, focusing on identifying discrepancies without altering the dataset. The primary action was the identification of CSV entries missing corresponding images. No imputation or row dropping was performed to maintain the dataset's original integrity.

### What Was Intentionally NOT Done

- **No Imputation:** Missing values were not imputed to avoid introducing bias.
- **No Row Dropping:** Rows with missing image files were not removed to preserve the dataset's completeness.
- **No Data Transformation:** The dataset was not transformed or normalized, ensuring that the original data distribution is retained for further analysis.

---

## 2. Unsupervised Exploration (L1)

### Bootstrap Findings

#### Image Distribution Analysis
- **Total Images**: 21,397
- **Dimensions**: All images are uniformly sized at 800x600 pixels.
- **Color Mode**: All images are in RGB mode.
- **Subdirectory Distribution**:
  - `train_images/`: 18,721 images
  - `test_images/`: 2,676 images

#### Tabular Metadata Analysis
- **Shape**: (18,721, 2)
- **Columns**:
  - `image_id`: 18,721 unique values, no missing data.
  - `label`: Low cardinality with 5 unique values, indicating potential categorical data.

#### Tabular Data Exploration
- **Label Distribution**:
  - Skewness: -1.161 (highly skewed)
  - Kurtosis: 0.899
  - Outliers: 939 (5.0%)
- **Correlation Analysis**: Not applicable due to insufficient numeric features.

### Domain Inference
The dataset likely pertains to an image classification domain, where images are categorized based on their content. The uniform image size and RGB mode suggest preprocessing for machine learning applications, possibly for training a classification model.

### Hypothesis Investigation Results

| Hypothesis ID | Hypothesis Description | Test Result | Verdict |
|---------------|------------------------|-------------|---------|
| H1            | The 'label' column represents different categories or classes of images. | Label distribution: {3: 61.55%, 4: 12.11%, 2: 11.17%, 1: 10.15%, 0: 5.02%} | CONFIRMED |
| H2            | The dataset is used for an image classification task. | Dataset structure and task description confirm classification use. | CONFIRMED |
| H3            | Images are uniformly sized and in RGB mode. | Images are uniformly sized and in RGB mode. | CONFIRMED |
| H4            | The dataset is split into training and testing subsets. | Train images: 18,721, Test images: 2,676. | CONFIRMED |
| H5            | The skewness in the 'label' distribution suggests class imbalance. | Label imbalance: {3: 61.55%, 4: 12.11%, 2: 11.17%, 1: 10.15%, 0: 5.02%} | CONFIRMED |

### Key Confirmed Findings and Implications
- **Uniform Image Preprocessing**: All images are consistently sized and in RGB mode, indicating preparation for machine learning tasks requiring standardized input.
- **Dataset Split**: The clear division into training and testing subsets supports typical machine learning workflows.
- **Class Imbalance**: The significant skew in label distribution suggests potential challenges in model training, necessitating strategies to address class imbalance for improved model performance.

---

## 3. Task-Aligned Feature Analysis (L2)

### Feature-Target Relationship Analysis

The task involves classifying cassava leaf images into one of four disease categories or a healthy category. The primary feature-target relationship is between the 'label' column and the image data. The 'label' column, containing 5 unique values, directly represents the target classes for classification. The consistent RGB color mode and dimensions of the images suggest that the dataset is well-prepared for image-based machine learning tasks. However, the class imbalance identified in the label distribution could impact model performance, particularly for underrepresented classes.

---

## 4. Recommendations

### 1. Advanced Feature Engineering Techniques

- **Data Augmentation**: To address class imbalance and enhance model robustness, apply data augmentation techniques such as random rotations, flips, zooms, and brightness adjustments. This can artificially increase the diversity of the training set and help the model generalize better.

- **Transfer Learning**: Utilize pre-trained models like EfficientNetV2 or Vision Transformers (ViT) as feature extractors. These models have been trained on large datasets and can provide a strong starting point for fine-tuning on the cassava leaf dataset.

- **Image Embeddings**: Consider generating image embeddings using a pre-trained model and then applying clustering techniques to explore potential sub-categories within the existing classes. This could reveal additional structure in the data that might be leveraged for improved classification.

### 2. Modern Model Architectures or Approaches

- **Vision Transformers (ViT)**: Given the uniform size and RGB nature of the images, Vision Transformers could be a suitable choice. They have shown state-of-the-art performance on image classification tasks and can handle large-scale datasets effectively.

- **EfficientNetV2**: This architecture is known for its efficiency and performance on image classification tasks. It can be particularly useful if computational resources are a concern, as it balances accuracy and efficiency.

- **Contrastive Learning**: Implement contrastive learning techniques such as SimCLR or BYOL to learn robust image representations. These methods can be particularly effective in scenarios with limited labeled data or class imbalance.

### 3. Better Preprocessing or Validation Strategies

- **Stratified Sampling**: Implement stratified sampling during the train-test split to ensure that each class is proportionally represented in both subsets, which can help mitigate the effects of class imbalance.

---

## 5. Review Notes

### Issues Fixed from the Strict Review

1. **Outlier Analysis Correction**: Clarified that the identified outliers in the `label` column are due to distribution issues like class imbalance, not values outside the expected range.
2. **Hypothesis H2 Reevaluation**: Confirmed the dataset's use for image classification based on its structure and task description, rather than the presence of labels in model scripts.
3. **File Count Consistency**: Ensured the total file count and breakdown are consistent and accurate.

### ML Suggestions Incorporated or Noted for Future Work

1. **Data Augmentation**: Recommended to address class imbalance and enhance model robustness.
2. **Transfer Learning and Modern Architectures**: Suggested using pre-trained models like EfficientNetV2 and Vision Transformers for improved classification performance.
3. **Contrastive Learning**: Proposed as a method to learn robust image representations, especially useful for class imbalance scenarios.
4. **Stratified Sampling**: Recommended for better train-test split to address class imbalance.

---