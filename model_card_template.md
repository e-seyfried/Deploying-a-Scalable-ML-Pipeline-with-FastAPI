# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a supervised binary classification model trained to predict whether an individual’s income is greater than $50,000 per year based on U.S. Census data. The model was implemented in Python using the scikit-learn library and trained using a Random Forest classifier. Data preprocessing included one-hot encoding of categorical variables and label binarization of the target variable. The trained model and preprocessor were serialized using pickle for later inference.

## Intended Use

The intended use of this model is to demonstrate a complete machine learning pipeline, including data processing, model training, evaluation, slice performance analysis, and API deployment. The model may be used for educational purposes but is not meant for real-world decision-making that affects individuals, such as hiring or lending.

## Training Data

The training data consists of U.S. Census income data stored in `census.csv`. The dataset contains demographic and employment-related features such as age, education, work class, occupation, marital status, race, sex, and native country. The data was split into training and testing sets using an 80/20 train-test split. The training portion was used to fit the model and the categorical encoders.

## Evaluation Data

The evaluation data consists of a 20% test set created from the same census dataset using a randomized split. This test dataset was not used during training and was only used for final model evaluation and for computing slice-based performance metrics across categorical features.

## Metrics

Model performance was evaluated using precision, recall, and F1 score. These metrics were chosen in order to balance false positives and false negatives in binary classification.

On the held-out test dataset, the model achieved the following overall performance:

- **Precision:** 0.7419  
- **Recall:** 0.6384  
- **F1 Score:** 0.6863  

In addition to overall evaluation, model performance was measured on slices of the data across categorical features. Slice-based performance varied significantly between groups. For example:

- For the **workclass = Private** slice (4,578 samples), the model achieved an F1 score of **0.6856**.
- For **workclass = Federal-gov** (191 samples), the F1 score was **0.7914**.
- For **workclass = Self-emp-not-inc** (498 samples), the F1 score dropped to **0.5789**.
- For **education = 10th** (183 samples), the F1 score was **0.2353**, indicating weak performance on this subgroup.
- For **education = 11th** (225 samples), the F1 score was **0.4286**.

These results show that model performance is not uniform across demographic and employment-based subgroups. All slice-level metrics were saved to `slice_output.txt`.

## Ethical Considerations

This model makes predictions about individual income using attributes such as education, sex, and race. Since these features correlate with historical and systemic inequalities between different demographic groups, the model may reflect existing biases in the data.

Slice-based evaluation revealed substantial differences in performance across groups. For example, the model performed substantially worse for lower education groups such as “10th” and “11th” education levels compared to higher education or government employment groups. These disparities raise concerns about fairness and potential harm if the model were applied to real-world decision-making.


## Caveats and Recommendations

The model was trained on a limited census dataset and may not generalize to populations outside this distribution. Slice-level evaluation shows that model performance varies considerably across demographic subgroups, particularly for groups with smaller sample sizes.

Future work should focus on improving model performance for underperforming subgroups, performing formal bias audits, and conducting hyperparameter tuning to improve overall robustness. Additional data collection and alternative feature representations may also help mitigate imbalances or disparities. Continuous monitoring of slice-based performance is recommended if the model is retrained or redeployed.
