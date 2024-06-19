# ClassificationThesholdTuner

ClassificationThesholdTuner is a tool to set the thresholds used for clasificiation problems and to visualize the implications of different thresholds. 

Where models produce probabilities for each class, depending on the distribution of probabilities and the metric we wish to optimize for, we may acheive better results using different thresholds. The tool automates selecting a threshold and helps you understand the choices related to the threshold(s) selected.

ClassificationThesholdTuner supports both binary and multi-class classification. In the binary classification case, it identifies a single threshold. For multi-class classification, it identifies a threshold for each class (other than the default class, described below).

Searching for an optimal threshold with binary classification problems is relatively straightforward, though ClassificationThesholdTuner does simplify the process. The use of visualizations also provides context. Optimizing the thresholds for multi-class classification is more complex. 

## Background
This assumes first, the use of classifiers that produce probabilities for each record and, second, that we wish to convert these into label predictions, such that for each record we have a prediction of a single class. 

Normally with classification problems (where we have a score for each record for each class), we would simply predict for each record the class that recieved the highest probability. In the binary classification case, this is equivalant to taking the class that received a predicted probability over 0.5 (though in very rare cases, both classes may recieve a probability of exactly 0.5 for some rows). In multi-class classification, there isn't a specific known split point, and the predicted class is simply the class with the highest estimated probability. 

In order to optimize certain metrics or to treat different classes differently (false positives or false negatives may be more relevant in some classes than others), it may be preferable to not follow the default behavior and to set a different threshold. It may be that certain types of errors are more signficant than others, and it is likely the case that the probabilities produced by the models are not well-calibrated (possibly even after specifically calibrating them in a post-processing step).

In cases where it's useful to predict a single class for each record, the relevant metrics will be based on the true and predicted labels, for example F1 Score, Matthews correlation coefficient, Kappa Score, or other such metric. These are all, in one way or another, derived from a confusion matrix. The confusion matrix, in turn, can look quite different depending on the thresholds used. 

All metrics can be misleading. They are single numbers (or a small set of numbers), and it's difficult to describe the quality of a model properly in a single number. When selecting a model (during tuning processes such as hyper-parameter tuning), it's necessary to use a single metric to select the best-performing model. But, when assessing a model and trying to get a sense of it's reliability, it's good to examine the output in multiple ways, including breaking it down by segment. A confusion matrix describes the output well for a given threshold (or, with multi-class problems, set of thresholds), but does not describe the model well given that a range of thresholds may potentially be used. 

The idea of this tool is to provide a fuller picture of the model's quality by examining a range potential thresholds, as well as selecting an optimal threshold (in terms of a specified metric).

## Thresholds in Binary Classification
Where there are only two classes, the models may output a probability for each class for each record. Scikit-learn, for example, works this way. However, one probability is simply 1.0 minus the other, so only the probabilities of one of the classes are strictly necessary. 

Normally, we would use 0.5 as the threshold, so the positive class would be predicted if it has a probability of 0.5 or higher, and the negative class otherwise. But, we can use other thresholds to adjust the behavior, allowing the model to be more, or less, eager to predict the positive class. For example, if a threshold of 0.3 is used, the positive class will be predicted if the model predicted the positive class with a probability of 0.3 or higher. So, compared to using 0.5 as the threshold, more predictions of the positive class will be made, increasing both false positives and true positives. 

## Thresholds in Multi-class Classification
Where we have multiple classes in the target column, if we wish to to set a threshold, it's necessary to also specify one of the classes as the default class. In many cases, this can be fairly natural. For example, if the target column represents medical conditions, the default class may be "No Issues" and the other classes may each relate to specific conditions. Or, if the data represents network logs and the target column relates to intrusion types, then the default may be "Normal Behavior" with the other classes each relating to specific network attacks.

In the example of network attacks, we may have a dataset with four distinct target values, with the target column containing the classes: "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For any record for which we run prediction, we will get a probability of each class, which will sum to 1.0. We may get, for example: 0.3, 0.4, 0.1, 0.2 (we assume here this orders the four classes as above). Normally, we would predict "Buffer Overflow" as this has the highest probability, 0.4. However, we can set a threshold in order to modify this behavior, which can affect the rate of false negatives and false positives for this class. 

We may specify, for example: the default class is 'Normal Behavior"; the threshold for "Buffer Overflow" is 0.5; for "Port Scan" is 0.55; and for "Phishing" is 0.45. By convention, the threshold for the default class is set to 0.0, as it does not actually use a threshold. So, the set of threhsolds here would be: 0.0, 0.5, 0.55, 0.45.

Then to make a prediction for any given record, we consider only the classes where the probability is over the relevant threshold. In this example (with predictions 0.3, 0.4, 0.1, 0.2), none of the probabilites are over their thresholds, so the default class, "Normal Behavior" is predicted.

If the predicted probabilities were: 0.1, 0.6, 0.2, 0.1, then we would predict "Buffer Overflow": the probability (0.6) is over its threshold (0.5) and this is the highest prediction.

If the predicted probabilites were: 0.1, 0.2, 0.7, 0.0, then we would predict "Port Scan": the probability (0.7) is over its threshold (0.55) and this is the highest prediction. 

If two or more classes have predicted probabilities over their threshold, we take the higher. If none do, we take the default class.

If the default class has the highest predicted probability, it will be predicted. 

## AUROC and F1 Scores
Numerous metrics are used for classification problems, but for simplicity, we will consider for the moment just two of the more common, the Area Under a Receiver Operator Curver (AUROC) and the F1 score (specifially, the macro score). These are often both useful, though measure two different things. AUROC measures how well-ordered the predictions are. It uses prediction scores (as opposed to labels) and applies only to binary prediction. Nevertheless, in a multi-class probablem, we can calculate the AUROC score for each class by treating the problem as a one-vs-all problem. For example, we can calculate the AUROC for each of "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For the AUROC for "Normal Behavior", we treat the problem as predicted "Normal Behavior" vs not "Normal Behavior", and so on.

In a binary classification problem, the AUROC evaluates how well the model tends to give higher probability predictions of the positive class to records of the positive class. That is, it looks at the rank order of the probabilities (not the actual probabilities). Other metrics, such as Brier Score and Log-Loss look at the probabilities themselves. These may be the most relevant metrics in some projects, though to optimize these, we often first work to ensure the probabilities are well-ranked (the AUROC is optimized), and then calibrate the model in post-processing. Alternatively, tuning the model to produce accurate probabilities in the first place is also common. 

The F1 Score, on the other hand, is able to work with multiple classes, and looks at the class predictions, not considering the probabilities behind them. Similarly for precision, recall, MCC (Matthew's Correlation Coeficient), and other metrics based on the predicted labels. Often, where label predictions are produced, models (behind the scenes) produce a score for each class and translate these scores to class predictions. 

To create good class predictions from probabilities, it's necessary only that the rank order of the probabilites is good. If the AUROC is high, it will be possible to create a set of class predictions with, for example, a high F1 score. 

However, it is also possible to have a high AUROC and low F1 Score, or a low AUROC and high F1 Score. The former is likely due to a poor choice of threshold. This can occur, for example, where the data is imbalanced and the AUROC curve hugs the y-axis. This can create an asymetric curve, where the optimal F1 score (or other such metrics) are found using threhsolds other than 0.5. 

The latter case (low AUROC score but high F1 Score) is likely due to a particuarly good choice of threshold (where most thresholds would perform poorly).

The AUROC is more straightforward when it has a standard, symetric shape, but this is not always the case. The AUROC averages over all possible thresholds. In practice, though, at least where we wish to produce labels as our predictions, we use only one threshold. The AUROC still useful metric, but it can be misleading if we use a sub-optimal threshold.  

## Adjusting the Threshold

An AUROC curve actually shows many potential threholds. Given an AUROC curve, as we go left and down, we are using a higher threshold. Less records will be predicted postive, so there will be both less true positives and less false positives. 

As we move right and up, we are using a lower threshold. More records will be predicted postive, so there will be both more true positives and more false positives. 

In the following plot, the red line represents a higher threshold than the green. 

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/img1.png)

The following presents a set of thresholds of a given AUROC curve. We can see where adjusting from one threhsold to another can affect the true postive and false postive rates to significantly different extents.

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/img2.png)

This is the main idea behind ajusting the threshold: it's often possible to achieve a large gain in one sense, while taking only a small loss in the other. 

## APIs
That explains the general idea. We now describe the ClassificationThesholdTuner itself, specifically the APIs it provides. 

The tool provides six APIs: 
- Three APIs (print_stats_labels(),  print_stats_table(), and print_stats_proba()) to assess the quality of the predictions given a set of ground truth values and predictions (either probabilities or class predictions). 
- Two APIs (plot_by_threshold() and describe_slices()) to visualize the implications of using different threhsolds
- One API (tune_threshold()) to optimize the threshold(s) for a specfied metric
- One API (get_predictions()) to get label predictions given a set of predictions and thresholds

## print_stats_labels()

This assumes the labels have been selected already. Where only the probabilities are available, this method cannot be called directly, but will be called by print_stats_proba() if that is called.

Display basic metrics related to the predictions. Displays some of the most common metrics: precision, recall and F1 scores. These are shown per class, as well as their macro average. 

This is method is called by print_stats_proba(), but can be called directly if the labels have been set elsewhere.

Example:
```python
tuner.print_stats_labels(
    y_true=d["Y"], 
    target_classes=target_classes,
    y_pred=d["Pred"])
```

Docstring:
```
Display basic metrics related to the predictions. This is method is called by print_stats_proba(), but can
be called directly if the labels have been set elsewhere.

y_true: array. 
    True labels for each record
target_classes: array. 
    Set of labels. Specified to ensure the output is presented in a sensible order.
y_pred: array. 
    Predicted labels for each record
return: None
```

Most methods have a parameter to specify the set of target labels. This is to specfiy the order of the probabilities passed in y_pred (which can be a 1d array in the binary classification case, but must be a 2d array in the multi-class classification case). It also ensures any plots are presented well, using the correct names of the target classes and an appropriate and consistent order for these. 

Example output (taken from one of the sample notebooks):
![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/binary_1.png)

## print_stats_table()
Presents a dataframe with a number of columns to step you through the operation. This can help point towards appropriate thresholds. 

Currently, this is only available for binary classification. 

gives stats of each range, and also the cumulative. If set a threshold, it's the cumulative that's relevant. 
1st two columns relate to the ranks of the probabilities. Due to ties, some ranges may have no records. 

Example:
```python
tuner.print_stats_table(
    y_true=d['Y'], 
    target_classes=target_classes,
    y_pred_proba=d["Pred_Proba"],
    num_ranges=10
)
```

Docstring:
```
Currently, this is available only for binary classification. It provides a breakdown of the precision and
recall (as well as other statistics) for each range of the predicted probabilities. For example, if 10 ranges
are used and y_pred_proba is evenly distributed from 0.0 to 1.0, then the first range will cover predicted
probabilities between 0.9 and 1.0; the next range will cover predicted probabilities between 0.8 and 0.9, and
so on.
For each range, this displays the precision (the number of positive records out of the total number of records
in this range of predicted probabilities) and the recall (the number of positive records in this range out of
the total number of postive examples).
The cumulative precision and cumulative recall for each range is also shown, giving insight into setting
the threshold at the low end of each range.
A plot of the cumulative precision and recall at each probability is also displayed.

y_true: array.
  True labels for each record
y_pred: array
  Predicted labels for each record
target_classes: array
  Set of labels. Specified to allow displaying the positive class by name.
num_ranges:
  The number of rows in the table displayed.

return: None
```

Example output:
![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/binary_3.png)

## print_stats_proba()
This presents basic statistics about the probabilities produced by a model. It presents 2 scores: brier score, AUROC,

It provides as plots: AUROC curve, histogram, and swarmplot. It enhances the AUROC plots to show the threshold. 

The plots help present the distribution of the predictions. If you have an AUROC score of, say, 0.91, this may be good. It means a random positive sample has a 91% chance of being ranked higher than random
negative sample. However, this does not provide the full story. If, for example, one class is in a small minority, the predictions may be clumped together, or spread more evenly. Knowing this helps set an appropriate threshold.

```python
tuner.print_stats_proba(
    y_true=d["Y"], 
    target_classes=target_classes, 
    y_pred_proba=d["Pred_Proba"]) 
```

Docstring:
```
Presents a summary of the quality of the predictions. This calls print_stats_labels() using the threshold
provided if any is; and using 0.5 otherwise.
This presents 2 scores related to the predicted probabilities: brier score, AUROC.
It also plots: AUROC curve, histogram, and swarm plot.

y_true: array
    True labels for each record
target_classes: A list of the unique values. Must include all values in y_true, though may include
    additional values not in y_true. In the case of binary classification where y_pred_proba contains only
    a series of values, we assume target_values contains first the negative class and second the positive class,
    with the probabilities representing the probabilities of the positive class.
y_pred_proba: If target_values contains 3 or more values, y_pred_proba must be a 2d matrix with a
    row for each record, each with a probability for each class. If target_values contains only 2 values,
    y_pred_proba may be in either this format or a single array of probabilities, representing the probability
    of the 2nd of the two values in target_value
thresholds: For binary classification should be a float. For multiclass classification, should be an
    array of floats, with a value for each class (the default class can be set to 0.0). If set to None, the
    default behaviour will be used to determine the class predictions.
return: None
```

Example output with binary classification:
![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/binary_2.png)

Example output with multi-class classification:
![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/multi_1.png)



## plot_by_threshold()
This draws a row of plots for each potential threshold. 
In the swarm plots, the red indicates errors. Otherwise, each class is drawn with a specific colour. 
The confusion matrix lines up with the swarm plots, showing the same classes in the same order. 

Example:
```python
tuner.plot_by_threshold(
    y_true=d['Y'], 
    target_classes=target_classes,
    y_pred_proba=d["Pred_Proba"])
```

Docstring:
```
Plot the effects of each of a range of threshold values. For multi-class classification, this uses the
same threshold for all classes -- it's simply to help understand the thresholds and not to tune them.

For each potential threshold, we draw a series of plots.

y_true: array. 
    True labels for each record
y_pred_proba: array. 
    Predicted labels for each record
target_classes: array. 
    Set of labels. Specified to ensure the output is presented in a sensible order.
start: float
    The first threshold considered
end: float
    The last threshold considered
num_steps: int 
    The number of thresholds
return: None
```

Example output:

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/binary_4.png)

## describe_slices()
For simplicity, assumes the same threshold for each class; this is for visualization and not the actual tuning. We can see the implications of moving the threshold from one location to another, in terms of the numbers of each class within each slice between the potential thresholds. 

Example:
```python
tuner.describe_slices(    
    y_true=d['Y'], 
    target_classes=target_classes,
    y_pred_proba=d["Pred_Proba"], 
    start=0.3, end=0.7, num_slices=5)
```

Docstring:
```
Give the count & fraction of each class within each slice. Currently, this feature is available only for
binary classification.

y_true: array.
    True labels for each record
:param target_classes: array.
    Set of labels. Specified to ensure the output is presented in a sensible order.
:param y_pred_proba: array.
    Predicted probability(ies) for each record
:param start: float
    Start of first slice considered
:param end: float
    End of last slice considered
:param num_slices: int
    Number of slices shown
:return: None
```

Example output:

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/binary_5.png)

## tune_threshold()
The main method. No output other than the threshold(s) found. In the case of binary classification, this returns a single threshold. In the case of multi-class classification, this returns a threshold per class. 

Can only take metrics based on labels, not probabilities. The goal of the tool is to find thresholds that maximize such metrics. 

Once called, may wish to call print_stats_labels() again with the optimal threshold(s). 

Example:
```python
from sklearn.metrics import f1_score

best_threshold = tuner.tune_threshold(
    y_true=d['Y'], 
    target_classes=target_classes,
    y_pred_proba=d["Pred_Proba"],
    metric=f1_score,
    average='macro',
    higher_is_better=True,
    max_iterations=5
)
print(best_threshold)
```

Docstring:
```
Find the ideal threshold(s) to optimize the specified metric.

y_true: array of str
    Ground truth labels for each record
target_classes: array of str
    List of unique values in the target column. Specifying this ensures they are displayed in a sensible order.
y_pred_proba: array of floats representing probabilities
    For multiclass classification, this is a 2d array. For binary classification, this may be 1d or 2d.
metric: function
    Must be a function that expects y_true and y_pred (as labels).
higher_is_better: bool
    For most most metrics (eg F1 score, MCC, precision), higher scores are better.
default_class: str
    Must be set for multiclass classification. Not used for binary classification.
max_iterations: The number of iterations affects the time required to determine the best threshold(s) and
    the precision of the threshold(s) returned.
kwargs: Any arguments related to the metric. For example, for f1_score, the average method may be
    specified.

return: For binary classification, returns a single threshold. For multi-class classification, returns a
    threshold for each class, with 0.0 set for the default class.
```


## get_predictions

Example:
```python
tuned_pred = tuner.get_predictions(
    target_classes=target_classes,
    d["Pred_Proba"], None, best_threshold)
```

Docstring:
```
Get the class predictions given a set of probabilities. 

target_classes: array of str
    Names of the target classes in the order of the probabilities in y_pred_proba
y_pred_proba: array of float 
    Predicted probabilities. For binary classification, may be a 1d or 2d array. For multiclass classification,
    must be a 2d array.            
default_class: str
    One element of target_classes
thresholds: float or array of float
    For binary classification, should be a float. For multiclass classification, must be an array of floats.

return: array of class labels
```

## Installation

This project uses a [single .py file](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/threshold_tuner.py).

This must be copied into your project and imported. For example:

```python
from threshold_tuner import ClassificationThesholdTuner

tuner = ClassificationThesholdTuner()
```


## Example Notebooks
Three example notebooks are provided. Two, for simplicity, use synthetic data:

[Binary Classification Example](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/notebooks/binary_classification_threshold_demo.ipynb)

and

[Multi-class Classification Example](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/notebooks/multiclass_classification_threshold_demo.ipynb)

As well, one is provided using a number of real datasets:

[Real Data Sets](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/notebooks/Real_Datasets.ipynb)


## Implications of setting the thresholds

There are some subtle points about setting thresholds in multiclass settings, which may or may not be relevant for any given project. 

A threshold can be over 0.5 or under. In this example, there are three classes: the default, B, and C, and the thresholds for both B and C are set to 0.7. This means any time a prediction for B is over its default, the prediction for B must be the highest. In the middle pane below we see a vertical line drawn at 0.7. Anything to the right will be predicted as B, though there are examples of the true class being each of the three classes. 

Similarly for the right pane: where the predicted probability of C is over its default of 0.7, this must be the highest predicted class, so anything to the right of the vertical line will be predicted as C. 

The default class works differently. Any prediction of 1.0 minus 0.7 (that is, 0.3) for the default class will result in a prediction of the default class. In this case, no other class can be over it's threshold. 

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/threshold_07.png)

The situation is more complicated, but similar where each class has a different threhsold. 

Where the threshold for each class is under 0.5, the situation may be different, as shown in the following plot. Here the thresholds for both B and C are set to 0.4. 

If there is a prediction for B under 0.4, it can't predict B. If there is a prediction between 0.4 and 0.5 for B, it may predict B. If there is a prediction over 0.5, it will predict B. So, in the middle pane, anything to the left of the first vertical line predicts Default or C, but not B. Anything between the two lines may predict any of the three classes. Anything to the right of the second vertical line will predict B. 

Similarly with respect to predictions for C, as shown in the third pane.

For the default, at 0.4, no other class can be above their threshold, so any predictions with a probability of 0.4 or higher for the default class will result in predictions of the default class. 

If the predicted probability for the default class is under 0.2, then even if the other two classes are tied, both are greater than or equal to their thresholds, so the higher of these (and not the default) will be predicted. 

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/threshold_03.png)






