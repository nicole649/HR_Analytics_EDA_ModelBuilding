Dataset from https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists/code?datasetId=1019790&sortBy=voteCount

Background:
A company provides data science courses for people from other companies and would like to hire data scientists among those people.

Task:
Predict if a candidate is looking for a job change or not.

Dataset:
Target is imbalanced. Null values in 7 features.

Challenge:
1. Target is imbalanced
There are lots of methods to deal with imbalanced targets. Some common methods are creating synthetic data or duplicating data. In this task, I used SMOTE/SMOTEENN/SMOTENC and resample, to see which perform better using Random Forest Classifier. Below are the feature importances from different upsampling methods:

![Upsampling_feature_importance](https://user-images.githubusercontent.com/88300660/134280523-4f51efc2-e78d-40e4-bc77-77e1b41f70f5.png)

It shows creating synthetic data changes the pattern of data and affects model prediction. While duplicating data does not change the model prediction.

Also, among the upsampling methods, SMOTE/SMOTEENN uses KNN technique to generate new data points in between two points. It creates values between 0 and 1, so there are values like 0.055, 0.21 for binary category after upsampling. There is no such issue by using SMOTENC and resample. One interesting thing is the model prediction tends to be better when using SMOTEENN.

2. Imputing missing values
As all the missing values are categorical labels, I replaced them by mode. The model prediction result is not satisfying.

![SMOTE_binary_data](https://user-images.githubusercontent.com/88300660/134280894-fa9e9347-3ec1-4cfd-880e-d3c49692018a.png)

Model used:
Random Forest, XGBoost, LightGBM and CatBoost are used in model building. Since XGBoost, LightGBM and CatBoost can handle imbalanced target and missing values, I chose to build models without upsampling and imputing. CatBoost can handle categorical features in its text form (no encoding needed), so I used it in the final model building.



Notebooks uploaded:
1.HR_Analytics_EDA
- data cleaning
- EDA
![experience_job](https://user-images.githubusercontent.com/88300660/134282423-f846b0bc-7637-4ade-b25d-cadde5751c48.png)
![correlation](https://user-images.githubusercontent.com/88300660/134282429-5efb6e96-b4db-4412-b055-cb6021b854cf.png)

2.HR_Analytics_Model_building
-Data preprocessing
-Model Building
![catboost](https://user-images.githubusercontent.com/88300660/134282588-2b8f415b-3980-4206-b67d-96b6d7a4448b.png)

3.HR Analytics_Upsampling
- Comparison of various upsampling methods

4.HR Analytics_LightGBM_XGBoost_CatBoost
- Comparison of different models
![AUC](https://user-images.githubusercontent.com/88300660/134283380-b2859fcb-c700-43b1-b6a0-3e7b9eca5888.png)


Techniques Used:
- Python
- Pandas
- SKlearn
- SMOTE
- XGBoost, LightGBM, CatBoost, RandomForest
- HPO tuning
