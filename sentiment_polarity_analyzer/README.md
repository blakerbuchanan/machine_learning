## Sentiment Polarity Analyzer Using Binary Logistic Regression

This directory contains code from a homework assignment wherein I implemented a sentiment polarity analyzer using binary logistic regression. The resulting algorithm takes as input a set of words and classifies the wordage as being positive (label = 1) or negative (label = 0).

This implementation uses ```feature.py``` to extract features from raw training, validation, and testing data. It then uses ```lr.py``` (logistic regression) to learn the parameters of a binary logistic regression model to classify the polarity of sets of words. A small data set has been provided for the purposes of testing. Larger datasets were used, but in the interest of storage, I have ommitted them from the directory.

### Running the feature extractor

To output formatted training, validation, and test data, run 

```python3 feature.py smalltrain_data.tsv smallvalid_data.tsv smalltest_data.tsv dict.txt formatted_train.tsv formatted_valid.tsv formatted_test.tsv 1```

in the terminal from the ```sentiment_polarity_analyzer``` directory.

### Running binary logistic regression on the formatted data

To run binary logistic regression on the resulting formatted data, run

```python3 lr.py formatted_train.tsv formatted_valid.tsv formatted_test.tsv dict.txt smalltrain_out.labels smalltest_out.labels smallmetrics_out.txt 60``` 

in the terminal from the ```sentiment_polarity_analyzer``` directory.

The ```small metrics_out.txt``` should show that the error rate on the training data is zero, while the error on the test data is around ```0.22```. 

### To do

1. Add explanation of the two different feature engineering models used (bag-of-word and trimmed) 
2. Add model derivation
3. Add curves of negative log-likelihood vs. epocs for training, validation, and testing data