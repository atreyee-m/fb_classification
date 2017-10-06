# To run the code

- run the script final_script.sh

# What it does:

 - It reads from the input file
 - It cleans up the text, e.g., replaces URLs with the word URL. There are other preprocessing steps which could be done such as removal of stop words, lemmatizing, etc. which I have not tried here.
 - Another question is what effect does the code mixed data (different languages mixed in the description field) have on the task in hand - this is an interesting question and needs to be addressed.
 - Next, it transforms the target (classes) into numeric value to be used by scikit-learn
 - Then the features are extracted
 - I have just used tf-idf weighting with bag of words as features. Other features (counts of POS tags, stopwords, specific keywords, topic usage using LDA)could be used to see if that improves the accuracy of the classification task.
 - I have used randomly split train and test classes
 - After this, I ran it through several classifiers - majority class (baseline, this is a dummy classifier which returns the base class), SVM (with linear kernel), random forests, gradient boosted trees, neural networks (multilayer perceptrons).

## Number of Datapoints
Fake Seller = 9174

Reseller    = 9583

No Seller   = 16425
 
Thus, the dataset is slightly imbalanced with "no seller" being the majority class.

 
## Accuracies:
 
| Classifier             | Accuracy           
| ---------------------- |:------------------:
| Majority               | 46.13%
| SVM               	 | 80.01%      
| Random Forest          | 81.65%      
| Gradient boosted trees | 73.94%
| Neural Networks        | 79.51%
 
Gradient boosted trees are based on xgboost and are usually good when it comes to speed and accuracy. However, in this case, random forest performs the best, followed by SVM. The parameters can be further tuned to improve the accuracy. I have not done any feature engineering in this case. However, numerous other features with proper feature selection (forward/backward works in most of the cases), can increase the accuracy further. I have reported the accuracy of the task, but there is scope for reporting precision/recall/confusion matrix.
Also, there is a scope for cross validation because this is the training dataset to tune the parameters further. There is a scope to accomodate for predefined train and test dataset.
Overall, the performance of the model depend largely on the datasets, hence a different kind of dataset might have different performance for the given models.
 