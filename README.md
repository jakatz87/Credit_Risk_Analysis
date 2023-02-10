# Credit Risk With Supervised Machine Learning

## Overview
I used the Sci-Kit and Imbalanced learning packages in Python to implement machine learning models on credit risk data to determine the likelihood of high-risk or low-risk loans.

## Resources
- Python with Scikit and Imbalanced Libraries
- Jupyter Notebook
- LoanStats_2019Q1.csv

## Process
Machine Learning (ML) can be simplified into model-fit-predict.  
Model: The data is separated into between 75% and 80% for training a model, the remainder is used for testing it.  
```
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y, random_state=1)
```
Fit: Many options are available for ML algorithms and how each works with the data.  Since the amounts for high-risk and low-risk loans were drastically different, I compared some models with Oversampling (increasing the amount of high-risk loans), Undersampling (lowering the amount of low-risk loans), a combination of both, and two separate models of ensemble learning that uses multiple amounts of models within each model.

Predict: Each model can predict its accuracy of the testing data (the remaining 20% - 25%) and the results are displayed using either a confusion matrix or a classification report.

NOTE: One feature of ML is the ability to measure the impact of each feature (variable) on the outcome of the prediction. Even though these results are only from one ML model, they are telling:
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Features.png)
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Features2.png)

The highest impacts on the predictions seem to be payback amounts, but only contribute to less than 8% of the model.  This report can lead to further modeling with fewer features.

## Results
I focused on the Classification Reports for each type of ML model, mainly on the Balanced Accuracy Score (the “geo” label) and the Harmonic Mean of the Precision and Sensitivity (the “f1” label)

Random Oversampler Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Random_Over1.png)
This model is quite precise with identifying low-risk loans, although not so well for high-risk loans which means a large number of false positives.  The sensitivity (“rec” label) is much better, meaning a reduction in the number of false negatives.  The Balanced Accuracy and F1 scores are not indicative of a good model.

SMOTE Oversampler Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/SMOTE1.png)
This model shares many features of the Random Oversampler, but the Balanced Accuracy and F1 scores are slightly higher.

Centroid Clusters Undersampler Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Cluster_Centroids1.png)
Although the precision and sensitivity scores are similar to the oversampler models, this model’s Balanced Accuracy and F1 scores are significantly lower.  This would be an intuitive result as lower sample sizes tend to reduce accuracy (as long as overfitting is not happening with the larger sample sizes).

SMOTEENN Combination Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/SMOTEENN1.png)
This model, although using both oversampling and undersampling has very similar results to both oversampling models.

Balanced Random Forest Bootstrap Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Balanced_Random_Forest1.png)
This model randomly undersamples each iteration of its decision tree, and was set to run 100 estimators.  The precision of low-risk loans was slightly higher than the other models, but the sensitivity is markedly improved.  This model is much less likely to generate false negatives for both high- and low-risk loans.  The Balanced Accuracy and F1 scores are much improved as well.

Easy Ensemble AdaBoost Model
![image](https://github.com/jakatz87/Credit_Risk_Analysis/blob/main/Resources/Easy_Ensemble_AdaBoost.png)
This model uses different bootstrap samples and uses random undersampling for balance, and was also set to run 100 estimators.  This model showed the highest amount of precision for high-risk loans, as well as the best Balanced Accuracy and F1 scores.

## Summary
Comparing all the models based on Balanced Accuracy and F1 scores should result in a clear choice of using the Easy Ensemble AdaBoost Model for a ML algorithm with this data set. The concern of overfitting data for this model is tempered by the fact that it uses random undersampling for each bootstrap.

I don’t recommend using any of these models, however.  The low precision scores for high-risk loans is quite troubling.  If high-risk loan labels have a high probability of false positives, these models may continue the decades-old lending practices that have shaped the socio economic landscape we are dealing with today.  False positive labels for high risk loans have made a drastic impact on geographical, racial, and economic tensions.  

More experimentation needs to be done using the previously mentioned feature importances and a wider variety of ML algorithms before any financial institution implements them.  


