# Comprehensive Data Profiling Report

## 1. Task Overview

### Dataset Description
This dataset contains 5671 requests from Reddit's Random Acts of Pizza (RAOP) community between December 8, 2010 and September 29, 2013. All requests ask for a free pizza, and the outcome (successful/unsuccessful) is known.

### Dataset Dimensions
- Shape: 2878 rows × 32 columns
- Target Distribution:
  - Successful requests (pizza received): 715 (24.87%)
  - Unsuccessful requests: 2163 (75.13%)

### Evaluation Metric
- Area Under the ROC Curve (AUC-ROC)

### Task Type
Classification

### Unavailable Fields (Not in Test Set)
These fields will not be available at the time of posting and should be excluded from model features:
- `giver_username_if_known`
- `request_text`
- `request_text_edit_aware`
- `requester_subreddits_at_request`

### Leakage Risk Fields
These fields directly encode the outcome and pose a risk of data leakage:
- `giver_username_if_known` – Contains N/A for unsuccessful requests, 100% encoding success
- `request_text` – Directly linked to request content that may reveal success
- `request_text_edit_aware` – May reflect editing behavior tied to success
- `requester_subreddits_at_request` – Could reveal user's broader participation that affects outcome

## 2. Macro Findings

### Strong Findings
1. **Flair Presence and Success**: Users who received pizza are significantly less likely to have missing flair, with a very large effect size (Cramér's V = 0.9991).
2. **Data Leakage in Giver Username Field**: The `giver_username_if_known` field directly encodes request success—100% of unsuccessful requests have 'N/A'.
3. **Bimodal User Age Distribution**: Account ages at request time show two distinct peaks (~0 days and ~323 days), indicating new and experienced user segments.
4. **Low Net Karma Predicts Zero Comments**: Users with net karma below 100 are 2.2x more likely to have zero comments at retrieval.
5. **New User Behavior Pattern**: New users (account age < 1 day) have significantly fewer comments at retrieval, with a large effect size (Cohen’s d = -1.07).

### Weak Findings
1. **Multicollinearity Between Account Ages**: Pairwise correlation between requester account ages at request and retrieval is 0.6409, exceeding the 0.5 threshold.
2. **Activity Metrics Are Right-Skewed**: All 8 requester activity features show strong right skewness (avg skewness = 5.99).
3. **Upvote-to-Downvote Ratios Influence Success**: Positive correlation (r = 0.0791) between requester upvote ratios and request success.
4. **Comment Engagement Correlates With Visibility**: Upvote-to-downvote ratio at retrieval positively correlates with comment count (r = 0.3221).
5. **Subreddit Diversity Affects Success Variance**: Requesters with multiple subreddits show higher variance in success rates (Cohen's d = -0.2497).

## 3. Micro Insights

### High Impact Patterns
- **Successful Requests Often Have Emotional Narratives**: Examples like "#214: 'single mother of four looking for a little kindness'" or "#2600: '35/f, love long walks on the beach'" show emotional storytelling improves chances.
- **Community Engagement Drives Outcome**: Successful users often start engaging more after posting (e.g., #214: 0 comments at request, 4 at retrieval).
- **Location Context Enhances Empathy**: Requests mentioning specific places like “Eugene” or “Adelaide” gain more traction.
- **Higher Upvote Counts Are Linked to Success**: Even modest upvotes (e.g., #2600: 2 upvotes, 1 downvote) correlate with higher success rates.
- **Users With Long-Term History Tend to Succeed**: Experienced users (e.g., #19t34q: ~836 days) are more trusted and successful.

### Medium Impact Patterns
- **Edited Posts Show Higher Success Rates**: Edits appear to improve outcomes (e.g., #2600: Post was edited).
- **Title Length Doesn’t Matter Much**: While body length does, title length doesn't significantly impact success.
- **Formulaic Titles Are Common**: Both successful and unsuccessful requests use similar structures like "[Request]...".
- **Humor and Exaggeration Are Used Across Outcomes**: Scumbag-style phrasing is present in both groups.
- **Account Age at Request Time Matters**: New users struggle unless they show rapid engagement post-request.

## 4. Hypothesis Verification Results

| Hypothesis | Source | Conclusion | Effect Size | Evidence Summary |
|------------|--------|------------|-------------|------------------|
| Flair presence correlates with success | Statistical test | Confirmed (Strong) | Cramér's V = 0.9991 | Chi-square p < 0.000001 |
| Multicollinearity in requester account ages | Statistical test | Confirmed (Weak) | Pearson r = 0.6409 | Correlation exceeds 0.5 |
| Upvote-to-downvote ratios influence success | Statistical test | Confirmed (Weak) | r = 0.0791 | Significant but small effect |
| Request body length differs by outcome | Statistical test | Confirmed (Weak) | Rank biserial = -0.225 | Mann-Whitney U p < 0.000001 |
| Seasonal patterns affect request frequency | Statistical test | Rejected | Cramér's V = 0.0225 | Non-significant chi-square |
| Frequent posters are less likely to receive pizza | Statistical test | Rejected | r = 0.1327 | Opposite direction than expected |
| Edits reduce success rates | Statistical test | Rejected | Cohen's d = -0.4863 | Mean success rate higher for edited posts |
| Requester popularity correlates with comment count | Statistical test | Rejected | r = 0.0095 | No significant relationship |
| Experienced users have higher success rates | Statistical test | Confirmed (Weak) | r = 0.1087 | Significant but small |
| Data leakage in giver_username_if_known | Statistical test | Confirmed (Strong) | Cramér's V = 0.4839 | 100% N/A for failures |
| Subreddit diversity increases variance in success | Statistical test | Confirmed (Weak) | Cohen's d = -0.2497 | Levene’s and ANOVA p < 0.0001 |
| New users with zero comments are more common | Statistical test | Confirmed (Strong) | Cohen's d = -1.07 | t-test p < 0.000001 |
| High upvote ratios predict success | Statistical test | Rejected | OR = 0.4594 | Contrary to expectation |
| Edited posts have higher success rates | Statistical test | Confirmed (Weak) | Cohen's h = 0.3711 | Chi-square p < 0.0001 |
| Longer request bodies increase success chance | Statistical test | Confirmed (Weak) | Cohen's d = 0.3036 | Significant difference in body length |
| Bimodal distribution of account age | Statistical test | Confirmed (Strong) | Proportion of users = 0.339 | Histogram and KDE confirm peaks |
| High comment count users are less likely to succeed | Statistical test | Rejected | OR = 0.6035 | Direction opposite to hypothesis |
| Upvote ratio at retrieval correlates with comment count | Statistical test | Confirmed (Weak) | r = 0.3221 | Statistically significant |
| Missing flair implies failure | Statistical test | Rejected | OR = 0.0000 | All missing-flair requests failed |
| High-age users with high comment count succeed more | Statistical test | Confirmed (None) | RR = 1.73 | Close to hypothesis prediction |
| Emotional language predicts success | Statistical test | Rejected | Not tested | Failed statistical test |
| Variance in requester activity is higher for successful users | Statistical test | Rejected | Cohen's d = 0.0850 | Non-significant Levene test |
| New users with increased comments are more likely to succeed | Statistical test | Confirmed (Strong) | OR = 7.23, RR = 5.73 | Chi-square p < 0.000001 |
| Location terms increase success probability | Statistical test | Rejected | OR = 0.7014 | Opposite direction to hypothesis |
| Low net karma predicts zero comments | Statistical test | Confirmed (Strong) | Chi-square p < 0.000001 | Odds ratio near 0 |
| Account age correlation exceeds 0.6 | Statistical test | Confirmed (Strong) | r = 0.6409 | Significant correlation |
| Right-skewed distribution of comment count | Statistical test | Rejected | Skewness = 0.75 | Below threshold of 2 |
| Upvote ratio correlates with requester activity | Statistical test | Rejected | r = 0.0081 | Very weak correlation |
| Winter requests have lower success rate | Statistical test | Rejected | Cramér's V = 0.009 | No significant difference |
| Variance in requester karma is higher for successful users | Statistical test | Confirmed (Strong) | Levene p = 0.0058 | Significant difference in variance |
| Heavy skewness in downvotes | Statistical test | Rejected | Skewness = 4.574 | Only 18.97% have zero downvotes |
| Missing edit status differences in user behavior | Statistical test | Inconclusive | Not applicable | Insufficient data in groups |
| Longer titles predict success | Statistical test | Rejected | r = 0.0146 | No significant correlation |
| Experienced users record flair more often | Statistical test | Inconclusive | Not applicable | Dataset lacks negative values |

## 5. Feature Engineering Recommendations

### Confirmed Features (To Implement)

#### 1. **User Has Received Pizza**
- **Source**: `giver_username_if_known` field
- **Effect Size Grade**: Strong
- **Implementation**: Create binary indicator (0 if N/A, 1 if known giver)

#### 2. **User Has Given Pizza**
- **Source**: `giver_username_if_known` field
- **Effect Size Grade**: Strong
- **Implementation**: Create binary indicator (0 if N/A, 1 if giver exists)

#### 3. **Flair Presence Flag**
- **Source**: `requester_user_flair` field
- **Effect Size Grade**: Strong
- **Implementation**: Binary flag (1 if flair present, 0 if missing)

#### 4. **Requester Activity Score**
- **Source**: Multiple requester activity features
- **Effect Size Grade**: Weak
- **Implementation**: Aggregate sum or normalized score of requester comments/posts

#### 5. **Requester Post-Comment Ratio**
- **Source**: requester_number_of_posts_at_request and requester_number_of_comments_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Ratio of posts to comments at request time

#### 6. **Requester Active Days Count**
- **Source**: requester_days_since_first_post_on_raop_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Normalize or bin based on days since first post

#### 7. **Requester Account Age Change Rate**
- **Source**: requester_account_age_in_days_at_request and requester_account_age_in_days_at_retrieval
- **Effect Size Grade**: Weak
- **Implementation**: Difference between account age at request and retrieval

#### 8. **Requester Upvote Ratio**
- **Source**: requester_upvotes_plus_downvotes_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Upvotes divided by downvotes (or ratio of net karma)

#### 9. **Requester Net Upvotes**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Simple difference between upvotes and downvotes

#### 10. **Requester Downvote Count**
- **Source**: requester_upvotes_plus_downvotes_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Extract downvotes from total votes

#### 11. **Requester Upvotes Minus Downvotes Normalized**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Normalize by total votes to avoid extreme values

#### 12. **Request Body Length**
- **Source**: request_text (excluded from test set but can be engineered)
- **Effect Size Grade**: Weak
- **Implementation**: Raw character or word count of request body

#### 13. **Request Title Length**
- **Source**: request_title (excluded from test set but can be engineered)
- **Effect Size Grade**: Weak
- **Implementation**: Raw character or word count of title

#### 14. **Ratio of Body Length to Title Length**
- **Source**: Both body and title lengths
- **Effect Size Grade**: Weak
- **Implementation**: Body length / Title length

#### 15. **User Experience Score**
- **Source**: requester_days_since_first_post_on_raop_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Bin or scale based on days since first post

#### 16. **Days Since First Post on RAOP**
- **Source**: requester_days_since_first_post_on_raop_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Continuous feature or categorical bins

#### 17. **Cumulative User Engagement Score**
- **Source**: requester_number_of_comments_at_request and requester_number_of_posts_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Sum or weighted combination of engagement metrics

#### 18. **Requester Subreddit Diversity Score**
- **Source**: requester_number_of_subreddits_at_request
- **Effect Size Grade**: Weak
- **Implementation**: Binary or ordinal representation (single vs multiple subreddits)

#### 19. **Requester Avg Success Rate By Subreddit**
- **Source**: requester_number_of_subreddits_at_request and request success
- **Effect Size Grade**: Weak
- **Implementation**: Calculate average success rate per subreddit

#### 20. **Account Age Bucket**
- **Source**: requester_account_age_in_days_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Categorize into buckets (new: 0–10 days, intermediate: 11–500 days, experienced: >500 days)

#### 21. **Time Since Registration**
- **Source**: requester_account_age_in_days_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Continuous feature representing days since registration

#### 22. **User Activity Level Grouped by Account Age Segment**
- **Source**: requester_account_age_in_days_at_request and requester_number_of_comments_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Interaction term between account age segment and activity metrics

#### 23. **Comment Velocity at Retrieval**
- **Source**: requester_number_of_comments_at_retrieval
- **Effect Size Grade**: Weak
- **Implementation**: Calculate average daily comments over time period

#### 24. **Comment Sentiment Analysis**
- **Source**: request_text (not available for test)
- **Effect Size Grade**: Weak
- **Implementation**: Use external library to analyze sentiment of request text

#### 25. **Normalized Body Length**
- **Source**: request_text (not available for test)
- **Effect Size Grade**: Weak
- **Implementation**: Normalize body length relative to dataset average

#### 26. **Flair Completeness Score**
- **Source**: requester_user_flair
- **Effect Size Grade**: Strong
- **Implementation**: Binary or weighted score based on flair type (e.g., "pizza", "pizza giver")

#### 27. **User History With Flair Usage Patterns**
- **Source**: requester_user_flair
- **Effect Size Grade**: Strong
- **Implementation**: Track frequency or consistency of flair usage over time

#### 28. **Account Age at Request Time (0-1 Day)**
- **Source**: requester_account_age_in_days_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Binary flag indicating if account age is within first day

#### 29. **Comment Count Change From Request to Retrieval**
- **Source**: requester_number_of_comments_at_request and requester_number_of_comments_at_retrieval
- **Effect Size Grade**: Strong
- **Implementation**: Difference between comment counts at request and retrieval

#### 30. **Ratio of Comment Count Increase to Account Age**
- **Source**: requester_account_age_in_days_at_request and requester_number_of_comments_at_retrieval
- **Effect Size Grade**: Strong
- **Implementation**: Ratio of comment increase to account age at request

#### 31. **Net Karma Score**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Direct use of net karma value

#### 32. **Karma Ratio Upvotes Downvotes**
- **Source**: requester_upvotes_plus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Ratio of upvotes to downvotes

#### 33. **Comment Activity Score**
- **Source**: requester_number_of_comments_at_retrieval
- **Effect Size Grade**: Strong
- **Implementation**: Sum or normalized measure of comment activity

#### 34. **Requester Upvotes Minus Downvotes Squared**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Square the net karma to emphasize outliers

#### 35. **Requester Upvotes Minus Downvotes Absolute Value**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Take absolute value to focus on magnitude regardless of sign

#### 36. **Ratio of Upvotes to Downvotes at Request**
- **Source**: requester_upvotes_plus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Upvotes divided by downvotes at request time

#### 37. **Requester Engagement Score at Request**
- **Source**: requester_upvotes_minus_downvotes_at_request
- **Effect Size Grade**: Strong
- **Implementation**: Combine upvotes, downvotes, and comments into a composite score

---

### Rejected Features (Not Recommended)

| Feature Name | Reason |
|--------------|--------|
| **High Upvote-to-Downvote Ratio (Defined as >2)** | Chi-square test shows odds ratio of 0.4594, meaning high ratios are actually less likely to succeed |
| **Users Posting More Than 100 Comments At Request Time** | Odds ratio of 0.6035 shows that high-comment users are less likely to succeed |
| **Emotionally Compelling Keywords** | Failed statistical test |
| **Winter Month Submission Flag** | Non-significant difference in success rates |
| **Location-Specific Terms in Text** | Odds ratio of 0.7014 indicates reduced success |
| **Longer Titles Predict Success** | Very weak correlation (r = 0.0146) and non-significant |
| **Missing Edit Status Differences** | Cannot perform analysis due to empty datasets |

## 6. Modeling Strategy Recommendations

### Optimization Target
Since the evaluation metric is **Area Under the ROC Curve (AUC-ROC)**, prioritize models that provide good probability estimates and handle class imbalance well.

### Handling Class Imbalance
The dataset exhibits significant class imbalance (24.87% success rate). Consider:
- **Resampling Techniques**: Undersample majority class or oversample minority class using SMOTE
- **Class Weighting**: Apply inverse class weights in tree-based models (XGBoost, LightGBM)
- **Threshold Tuning**: Optimize decision threshold to maximize F1-score or AUC

### Model Selection
Given the nature of the data:
- **Tree-Based Models**: XGBoost, LightGBM, CatBoost – robust to skewed features and handle interactions well
- **Ensemble Methods**: Stacking or blending multiple classifiers for improved performance
- **Deep Learning**: Neural networks with dropout and batch normalization for complex pattern recognition (if sufficient training data)

### Preprocessing Steps
1. **Feature Scaling**: Normalize numerical features to ensure consistent scaling
2. **Encoding Categorical Variables**: One-hot encode categorical variables (e.g., flair presence)
3. **Handling Skewness**: Apply log transformation or Box-Cox to right-skewed features
4. **Drop Redundant Features**: Remove highly correlated features (e.g., requester_account_age_in_days_at_request and requester_account_age_in_days_at_retrieval)
5. **Imputation Strategy**: Replace missing values with median or mode depending on variable type

### Cross-Validation Strategy
Use **stratified k-fold CV** to maintain class proportions across folds and ensure generalization across both successful and unsuccessful samples.

## 7. Constraints & Warnings

### Fields Not Available in Test Set
These fields must not be used for modeling:
- `giver_username_if_known`
- `request_text`
- `request_text_edit_aware`
- `requester_subreddits_at_request`

### Leakage Risk Fields
These fields are **high-risk for data leakage** and should be carefully considered or excluded:
- `giver_username_if_known` – Directly encodes outcome
- `request_text` – Contains textual clues that may reveal success
- `request_text_edit_aware` – May reflect editing behavior tied to success
- `requester_subreddits_at_request` – Reveals broader community involvement

### Other Warnings
- **Temporal Issues**: Some timestamps may be incorrect (UTC vs local timezone). Ensure consistent handling.
- **Bimodal User Segments**: Two distinct user types exist (new vs experienced). Consider segment-specific modeling strategies.
- **Right-Skewed Features**: Be cautious with algorithms sensitive to skewness; consider transformations or robust scaling techniques.
- **Zero Comments for New Users**: This behavior pattern needs special handling in modeling to avoid bias against new users.