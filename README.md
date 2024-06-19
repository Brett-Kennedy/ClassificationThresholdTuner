# ClassificationThesholdTuner

ClassificationThesholdTuner is a tool to set the thresholds used for clasificiation problems and to visualize the implications of different thresholds. 

Where models produce probabilities for each class, depending on distribution of probabilities and the metric we wish to optimize for, we may acheive better results using different thresholds. The tool automates selecting a threshold and helps understand the choices related to the thresholds selected.

It supports both binary and multi-class classification. In the binary classification case, it identifies a single threshold. For multi-class classification, it identifies a threshold for each class (other than the default class, described below).

Searching for an optimal threshold with binary classification problems is relatively straightforward, though this does simplify the process. The use of visualizations also provides context. Optimizing the thresholds for multi-class classification is more complex. 

## Background
This assumes the use of classifiers that produce probabilities for each record and that we wish to convert these into label predictions, so that for each record we have a prediction of a single class. Normally we would simply predict the class that recieved the highest probability. In the binary classification case, this is equivalant to taking the class that received a predicted probability over 0.5 (though in very rare cases, both classes may recieve a probability of exactly 0.5 for some rows). In multi-class classification, there isn't a specific known split point, and the predicted class is simply the class with the highest estimated probability. 

In order to optimize certain metrics or to treat different classes differently, it may be preferable to not follow the default behavior and to set a different threshold. It may be that certain types of errors are more signficant than others, and it is likely the case that the probabilities produced by the models are not well-calibrated (possibly even after specifically calibrating them in a post-processing step).

In cases where it's useful to predict a single class for each record, the relevant metrics will be based on the true and predicted labels, for example F1 Score, Matthews correlation coefficient, Kappa Score, and so on. These are all, in one way or another, derived from a confusion matrix. The confusion matrix, in turn, can look quite different depending on the thresholds used. 

All metrics can be misleading. They are single numbers (or a small set of numbers), and it's difficult to describe the quality of a model properly in a single number. When selecting a model (during tuning processes such as hyper-parameter tuning), it's necessary to use a single metric to select the best-performing model. But, when assessing a model and trying to get a sense of it's reliability, it's good to examine the output in multiple ways, including breaking it down by segment. A confusion matrix describes the output well for a given threshold (or set of thresholds), but does not describe the model well given that a range of thresholds may potentially be used. 

The idea of this tool is to provide a fuller picture of the model's quality by examining a range potential thresholds, as well as selecting an optimal threshold (in terms of a specified metric).

## Thresholds in Binary Classification
Where there are only two classes, the models may output a probability for each class for each record. Scikit-learn, for example, works this way. But, one probability is simply 1.0 minus the other, so only the probabilities of one of the classes are strictly necessary. 

Normally, we would use 0.5 as the threshold, so the positive class would be predicted if it has a probability of 0.5 or higher, and the negative class otherwise. But, we can use other thresholds to adjust the behavior, allowing the model to be more, or less, eager to predict the positive class. For example, if a threshold of 0.3 is used, the positive class will be predicted if the model predicted the positive class with a probability of 0.3 or higher. So, compared to using 0.5 as the threshold, more predictions of the positive class will be made, increasing both false positives and true positives. 

## Thresholds in Multi-class Classification
Where we have multiple classes in the target column, if we wish to to set a threshold, it's necessary to also specify one of the classes as the default class. In many cases, this can be fairly natural. For example, if the target column represents medical conditions, the default class may be "No Issues" and the other classes may each relate to specific conditions. Or, if the data represents network logs and the target column relates to intrusion types, then the default may be "Normal Behavior" with the other classes each relating to specific network attacks.

In the case of network attacks, we may have a dataset with four distinct target values, with the target column containing the classes: "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For any record for which we run prediction, we will get a probability of each class, which will sum to 1.0. We may get, for example: 0.3, 0.4, 0.1, 0.2 (we assume here this orders the four classes as above). Normally, we would predict "Buffer Overflow" as this has the highest probability, 0.4. However, we can set a threshold in order to modify this behavior, which can affect the rate of false negatives and false positives for this class. 

We may specify, for example: the default class is 'Normal Behavior", the threshold for "Buffer Overflow" is 0.5, for "Port Scan" is 0.55 and for "Phishing" is 0.45. By convention, the threshold for the default class is set to 0.0, as it doesn't use a threshold. So, the set of threhsolds here would be: 0.0, 0.5, 0.55, 0.45.

Then to make a prediction for any given record, we consider only the classes where the probability is over the relevant threshold. In this example, none of the probabilites are over their thresholds, so the default class, "Normal Behavior" is predicted.

If the predicted probabilities were: 0.1, 0.6, 0.2, 0.1, then we would predict "Buffer Overflow": the probability (0.6) is over its threshold (0.5) and this is the highest prediction.

If the predicted probabilites were: 0.1, 0.2, 0.7, 0.0, then we would predict "Port Scan": the probability (0.7) is over its threshold (0.55) and this is the highest prediction. 

If two or more classes have predicted probabilities over their threshold, we take the higher. If none do, we take the default class.

If the default class has the highest predicted probability, it will be predicted. 

## AUROC and F1 Scores
Numerous metrics are used for classification problems, but for simplicity, we will consider for the moment just two of the more common, the Area Under a Receiver Operator Curver (AUROC) and the F1 score (specifially, the macro score). These are often both useful, though measure two different things. AUROC measures how well-ordered the predictions are. It uses prediction scores and applies only to binary prediction. Nevertheless, in a multi-class probablem, we can calculate the AUROC score for each class by treating the problem as a one-vs-all problem. For example, we can calculate the AUROC for each of "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For the AUROC for "Normal Behavior", we treat the problem as predicted "Normal Behavior" vs not "Normal Behavior", and so on.

In a binary classification problem, the AUROC evaluates how well the model tends to give higher probability predictions of the positive class to records of the positive class. That is, it looks at the rank order of the probabilities (not the actual probabilities). Other metrics, such as Brier Score and Log-Loss look at the probabilities themselves. These may be the most relevant metrics in some projects, though to optimize these, we often first work to ensure the probabilities are well-ranked (the AUROC is optimized), and then calibrate the model in post-processing. Alternatively, tuning the model to produce accurate probabilities in the first place is also common. 

The F1 Score, on the other hand, looks at the class predictions, not considering the probabilities behind them. Similarly for precision, recall, MCC (Mathew's Correlation Coeficient), and other metrics based on the predicted labels. Often models produce a score for each class and translate these scores to class predictions. 

To create class predictions from probabilities, it's necessary only that the rank order of the probabilites is good. If the AUROC is high, it will be possible to create a set of class predictions with, for example, a high F1 score. 

However, it's also possible to have a high AUROC and low F1 Score, or a high F1 Score and low AUROC. The former is likely due to a poor choice of threshold. This can occur, for example, where the data is imbalanced and the AUROC curve hugs the y-axis. This can create an asymetric curve, where the optimal F1 score (or other such metrics) are found using threhsolds other than 0.5. 

The latter case (low AUROC score but high F1 Score) is likely due to a particuarly good choice of threshold (where most thresholds would perform poorly).

The AUROC is more straightforward when it has a standard, symetric shape, but this is not always the case. The AUROC averages over all possible thresholds. In practice, though, at least where we wish to produce labels as our predictions, we use only one threshold. The AUROC still useful metric, but it can be misleading if we use a sub-optimal threshold.  

## Adjusting the Threshold

Given an AUROC curve, as we go left and down, we are using a higher threshold. Less records will be predicted postive, so there will be both less true positives and less false positives. 

As we move right and up, we are using a lower threshold. More records will be predicted postive, so there will be both more true positives and more false positives. 

Red, in this plot, is a higher threshold than green. 

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/img1.png)

The following presents a set of thresholds of a given AUROC curve. We can see where adjusting from one threhsold to another can affect the true postive and false postive rates to significantly different extents.

![Line Graph](https://github.com/Brett-Kennedy/ClassificationThresholdTuner/blob/main/images/img2.png)

## APIs
The tool provides six APIs: 
- three APIs (print_stats_labels(),  print_stats_table(), and print_stats_proba()) to assess the quality of the predictions given a set of ground truth values and predictions (either probabilities or class predictions). 
- two APIs (plot_by_threshold() and describe_slices()) to visualize the implications of using different threhsolds
- one API (tune_threshold()) to optimize the threshold(s) for a specfied metric
- one API (get_predictions()) to get label predictions given a set of predictions and thresholds

If have AUROC of, say, 0.91, may be good. But, just means random positive sample 91% change ranked higher than random
negative sample. If are a small minority, can be clumped together, or spread more evenly. Knowing helps set threshold.

methods specify target_labels to ensure handle in a sensible order. Especially important with binary classification,
so knows which the probabilities refer to where only one set of probs are provided (can provide a pair of probs
per row if want -- and for multiclass, need to).

### print_stats_labels()
This assumes the labels have been selected already. Where only the probabilities are available, this method cannot be called directly, but will be called by print_stats_proba() if that is called.

Display basic metrics related to the predictions. Displays some of the most common metrics: precision, recall and F1 scores. These are shown per class, as well as their macro average. 

This is method is called by print_stats_proba(), but can be called directly if the labels have been set elsewhere.

### print_stats_table()
Gives a number of columns to step you through the operation. Can point towards thresholds. 
currently only available for binary classification. 
gives stats of each range, and also the cumulative. If set a threshold, it's the cumulative that's relevant. 
1st two columns relate to the ranks of the probabilities. Due to ties, some ranges may have no records. 

### print_stats_proba()

Presents 2 scores: brier score, AUROC,
Plots: AUROC curve, histogram, and swarmplot.
If no threshold is specified, uses 0.5. Enhances the AUROC to show the threshold. 

### plot_by_threshold()
This draws a row of plots for each potential threshold. 
In the swarm plots, the red indicates errors. Otherwise, each class is drawn with a specific colour. 
The confusion matrix lines up with the swarm plots, showing the same classes in the same order. 

### describe_slices()
For simplicity, assumes the same threshold for each class; this is for visualization and not the actual tuning. We can see the implications of moving the threshold from one location to another, in terms of the numbers of each class within each slice between the potential thresholds. 

### tune_threshold()
The main method. No output other than the threshold(s) found. In the case of binary classification, this returns a single threshold. In the case of multi-class classification, this returns a threshold per class. 

Can only take metrics based on labels, not probabilities. The goal of the tool is to find thresholds that maximize such metrics. 


Once called, may wish to call print_stats_labels() again with the optimal threshold(s). 

# Examples
-- give examples from the notebooks

## Implications of setting the thresholds
-- give examples from notebooks

include some examples where AUROC is high and F1 is low. I think when this happens, the threshold can just be split 
better. I think get this when imbalance and the model mostly predicts neg as neg, so the curve line hugs the y axis.

give examples with real models on real data. can mostly use RF, but also sklearn MVP, DT, kNN.

In multi-class case, if move the threshold, we need to define a default class. Often this makes sense. eg if network
logs, default is not-an-attack, and all other classes types of attack. Or if medical, default is healthy and other 
classes types of condition. Then, only take each class (other than default) if it's both the highest and over the 
threshold. For now, have 1 threshold for all classes. Future versions will support separate threshold for each. 

if the threshold is, say, 0.6, and the probability for class B is 0.5, then goes to default. If prob is 0.65, then 
predict class B. So long as the default is over 0.5, it's fairly straightforwrd: if any class is over the threshold,
then no other class can be too. But if threshold is under, say, 0.4, then can have 2 classes over threshold. In that
case, go with the highest. Go with the default if either: it's the highest prob; none are over the threshold.

when plotting with multiclass, plot prob vs actual class, but are a set of probs per target class, so that many plots.
More effort, but get a fuller picture. 

evidently also a good tool. i guess. it doesn't render for me on bns laptop

It only makes sense to set a threshold with multiple classes if there is a default class. If there is no class over the
threshold and there is no default, then have no predition (in that case, 'no prediction' becomes a defacto default). 




