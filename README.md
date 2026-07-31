# Predicting Medical Insurance Costs

## Project Overview
This project presents a machine learning workflow for predicting medical insurance charges using demographic, lifestyle, and health-related data from the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) .<br>
The analysis includes data validation, exploratory data analysis, feature engineering, model development, hyperparameter tuning, model evaluation, and explainable AI using SHAP to interpret model predictions.<br>

---
## Problem statement
It is very important for insurance and healthcare organizations to accurately estimate medical insurance costs<br>
because doing so supports:-
- Premium estimation
- Pricing strategies
- Customer segmentation
- Financial planning
- Risk assessment<br>
The main objective of this project is to determine whether medical insurance costs can be predicted given an<br>
individual's demographic and lifestyle characteristics.

---
## Dataset
The data was sourced from Kaggle.<br>
It contains 1,338 records that describe policyholders through:
| Variable | Description                                          |
| -------- | ---------------------------------------------------- |
| Age      | Age of the primary policyholder                       |
| Sex      | Gender of the policyholder                            |
| BMI      | Body Mass Index                                      |
| Children | Number of dependents covered by the insurance policy |
| Smoker   | Smoking status                                       |
| Region   | Residential region in the United States              |
| Charges  | Individual medical insurance costs (Target Variable) |

---
## Project Workflow
**1. Data Validation**<br>
Dataset inspection
Data type verification
Missing value assessment
Duplicate record detection

**2. Data Cleaning**<br>
Performed necessary data cleaning to improve data quality before analysis and modeling.

**3. Feature Engineering**<br>
Additional variables were specifically created to support exploratory data analysis and<br>
were not used during model training.

**4. Exploratory Data Analysis**<br>
**Univariate Analysis**<br>
Explored the distribution of:
- Age
- BMI
- Gender
- Smoking Status
- Insurance Charges

**Bivariate Analysis**<br>
Investigated relationships between:
- Region vs Charges
- BMI Category vs Charges
- Smoking Status vs Charges
- Age Group vs Charges
- Gender vs Charges
- Region vs Age Group
- Number of Children vs Charges
- BMI Category vs Age Group

**Multivariate Analysis**<br>
Examined complex interactions among variables using:
- Age Group, Smoker, and Charges
- Region, Smoker, and Charges
- BMI Category, Smoker, and Charges
- Distribution of Numerical Variables
- Correlation Analysis
- Pairwise Numerical Relationships
- Distribution of Categorical Variables

**5. Data Preprocessing**
- Feature-target split
- Train-test split
- Feature scaling (where required)
- Categorical variable encoding
- Native categorical handling for CatBoost

**6. Baseline Model Development**<br>
The following regression models were trained and compared:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor

Evaluation metrics included:<br>
MAE  - Mean Absolute Error<br>
MSE  - Mean Squared Error<br>
RMSE - Root Mean Squared Error<br>
MAPE - Mean Absolute Percentage Error<br>
R² Score - Coefficient of Determination<br>

**7. Hyperparameter Tuning**<br>
Hyperparameter tuning was performed using cross-validation to improve model performance.
Each model was assigned an appropriate hyperparameter search space before selecting the optimal configuration.

**8. Model Comparison**<br>
Baseline and tuned models were compared to evaluate the effectiveness of hyperparameter optimization.

**9. Feature Importance**<br>
The final model's feature importance was analyzed to determine which variables contributed most to prediction accuracy.

**10. Model Explainability**<br>
SHAP was used to explain both global and local model behaviour through:
- SHAP Feature Importance
- SHAP Beeswarm Plot
- SHAP Waterfall Plot

These visualizations explain:<br>
- which features are most influential,
- whether they increase or decrease predictions,
- and why a particular prediction was made.

**11. Model Evaluation**<br>
The final model was evaluated using:
- Residual Plot
- Actual vs Predicted Plot
These diagnostics were used to assess prediction accuracy and identify potential model bias.

---
## Technologies Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- SHAP

---
## Key Findings
- Smoking status is the strongest predictor of medical insurance charges.
- BMI and age also contribute substantially to insurance cost prediction.
- Tree-based ensemble models outperform traditional linear regression models.
- Hyperparameter tuning improved model performance across all ensemble methods.
- SHAP analysis provided interpretable explanations for both individual predictions and overall model behaviour

---
## Business Recommendations
➤ **Conduct Anti-Smoking Campaigns**<br>
From EDA outcomes and feature importance chart, smoking status was found to be the major factor that influences<br>
medical insurance charges. SHAP analysis showed that being a smoker significantly increases one's medical insurance<br>
charges, while being a non-smoker incurred lower medical insurance charges.<br>
Seeing as smokers pose a risk to insurers, it is best to promote anti-smoking campaigns in order to cut the risks<br>
associated with policies for smokers by offering lower premiums or other rewards to people who quit smoking for<br>
good could push more policyholders to actually follow through. with it.<br>
This campaign could potentially lower the number of policyholders who are smokers, and also tend to reduce<br>
medical insurance charges in future.<br>

➤ **Encourage Weight Management**<br>
Body Mass Index (BMI) was the second strongest predictor of medical insurance cost.<br>
The data showed the presence of more obese people than any other BMI category. This is an indication that<br>
that there probably are more individuals in the obese BMI category than others. Higher BMI values were linked <br>
to higher medical insurance charges.<br>
In order to help policyholders manage their medical costs, health and wellness programmes that target proper<br>
nutrition, exercising regularly, and other healthy lifestyles should be encouraged. <br>
with the aim of helping them bring their weight under control.<br>

➤ **Prioritize High-Risk Customer Segments**<br>
The analysis revealed that policyholders who smoke, have a higher BMI, and are older tend to incur higher medical<br>
insurance charges. These risk factors, particularly when they occur together, are associated with increased healthcare costs.<br>
It is recommended that policyholders with multiple risk factors be identified and provided with targeted preventive<br>
healthcare programs, regular health screenings, and personalized wellness initiatives to help manage their long-term health.<br>

➤ **Continuously Monitor Model Performance**<br>
While the model achieved strong predictive performance, the residual analysis showed that a small number of high-cost<br>
policyholders remained difficult to predict accurately, suggesting that some important factors were not captured in the dataset.<br>
It is recommended to regularly retrain the model using updated claims data and incorporate additional features, such as<br>
chronic conditions, medical history, and healthcare utilization, to improve predictions for high-cost cases.<br>
Keeping the model current and enriching it with more comprehensive data can improve prediction accuracy over time and support better pricing and risk management decisions.<br>

---
## Visualizations

### Distribution of Charges

![Insurance Charges](images/Charges.jpg)

---
### Smoking Status vs Insurance Charges

![Smoker vs Charges](images/Smoker_Charges.jpg)

---
### Actual vs Predicted Values

![Actual vs Predicted](images/Actual_Predicted.jpg)

---
### Feature Importance

![Feature Importance](images/Feature_Importance.jpg)

---
### SHAP Summary
![SHAP Summary](images/SHAP.jpg)

---
## Prerequisites
Before running this project, ensure you have the following installed:
- Python 3.11 or later
- Git
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

Clone the repository:

```bash
git clone https://github.com/pg82647/Predicting_Medical_Insurance_Cost.git
cd Predicting_Medical_Insurance_Cost
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Open the notebook located in the `notebook/` directory and run the cells sequentially.


---
## Repository Structure

```text
medical-insurance-cost-prediction/
├── data/
│   └── insurance.csv
├── images/
│   ├── Actual_Predicted.jpg
│   ├── Charges.jpg
│   ├── Feature_Importance.jpg
│   ├── SHAP.jpg
│   └── Smoker_Charges.jpg
├── models/
├── notebook/
│   └── modeling_medical_insurance_cost.ipynb
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---
## Future Improvements
Future work could improve the quality and practical value of this project by:<br>
* While the best model did perform well in predicting the target, even though the model performed well, suggests that<br>
there may be several other features not captured in the dataset. Adding more features would greatly improve future projects.
* It is worth noting that the dataset used contained a very small number of records. It would help future projects if<br>
they could include, not just more features, but also more diverse observations.<br>
* As technology continues to advance, so do new and more advanced algorithms. Perhaps applying more advanced algorithms<br>
may further improve performance in future projects.
* While metrics such as R², RMSE, and MAE provide a good measure of model performance, they do not capture the real-world<br>
cost of prediction errors. Future projects could evaluate the financial impact of underestimating or overestimating<br>
insurance charges, offering a clearer picture of the model's usefulness for pricing policies and managing risk.

---
