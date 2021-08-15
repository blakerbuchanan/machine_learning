## Hidden Markov Models and Part-of-speech Tagging

This directory contains code for performing part-of-speech tagging using the forward-backward algorithm on a hidden Markov model.

### Part-of-speech tagging

Part-of-speech tagging is the process of labeling words in a set of text using a category of labels. For instance, the sentence

*Blake enjoys hiking.*, 

can be labeled as follows. Blake (noun) enjoys (verb) hiking (noun) . (punctuation)

There exist a variety of ways to label such sets of text. The data sets included in this directory were part of an assignment for an ML course at Carnegie Mellon and include tags like location, organization, person, or miscellanous, and common words taken from a news article.

### Running learnhmm.py and forwardbackward.py

To perform part-of-speech tagging using the forward-backward algorithm on a hidden Markov model for a set of 1000 words, first run

```
python learnhmm.py trainwords.txt index_to_word.txt index_to_tag.txt hmmprior_full_1000.txt hmmemit_full_1000.txt hmmtrans_full_1000.txt
```

to learn the parameters of the hidden Markov model. This will output text files to learn the initialization probabilities, transition probabilities, and emission probabilities.

Afterward, run

```
python forwardbackward.py testwords.txt index_to_word.txt index_to_tag.txt hmmprior_full_1000.txt hmmemit_full_1000.txt hmmtrans_full_1000.txt predicted.txt metrics.txt
```

to run the forward-backward algorithm and do prediction on the test data. Open the metrics.txt file to check the accuracy.

While this code works, note that it has not been optimized at all and is largely a brute-force implementation of this method.