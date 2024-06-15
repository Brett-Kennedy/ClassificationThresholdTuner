# ClassificationThesholdTuner

ClassificationThesholdTuner is a tool to visualize and to help set the thresholds used for clasificiation problems. Where models produce probabilities for each class, depending on distribution of probabilities and the metric we wish to optimize for, we may acheive better results using different thresholds. The tools automates selecting a threshold and helps understand the choices related to the thresholds selected.

We assume the use of classifiers that produce probabilities for each record and assume that we wish to convert these into label predictions, so that for each record we have a prediction of a single class. Normally we would simply predict the class that recieved the highest probability. In the binary classification case, this is equivalant to taking the class that received a predicted probability over 0.5 (though in very rare cases, both classes may recieve a probability of exactly 0.5 for some rows). In multi-class classification, there isn't a specific known point, and is simply the class with the highest estimated probability. 

It may be preferable to not follow the default behavior and to set a different threshold, in order to optimize certain metrics or to treat different classes differently. It may be that certain types of errors are more signficant than others, and it is likely the case that the probabilities produced by the models are not well-calibrated (possibly even after specifically calibrating them in a post-processing step).

The tool may be used for both binary classification and multi-class classification, though is simpler in the binary case. With binary probablems, only a single threshold must be found. With multi-class classification, the tool seeks to identify the optimal threshold for each class. As the two cases are somewhat different, we discuss them separately next.

If binary labels are what's relevant, can be productive to adjust the thresholds. The confusion matrixes can look quite different. 

-- idea isn't just to tune the threshold(s) but to be able to understand them too. Though, can simply tune as well, where the details are not relevant. 

all metrics can be misleading. they are single numbers (or a small set of numbers). Can view otherwise. Confusion matrix can explain fairly well, but it does hide a lot -- it's based on a certain threshold (or mulitple if multiple classes), so does not show how would look with other choices for the threshold. When selecting a model, it's necessary to use a single metric (though it can be a complex combination of multiple metrics -- it still needs to be a single number). But when assessing the model, can use multiple metrics. Gives a fuller picture. Good to break down by segment, for example. 

## Thresholds in binary classification
Where there are only two classes, the models may output a probability for each class for each record. Scikit-learn, for example, works this way. But, one probability is simply 1.0 minus the other, so only the probabilities of one of the classes are strictly necessary. 

Normally, we would use 0.5 as the threshold, so the positive class would be predicted if it has a probability of 0.5 or higher, and the negative class otherwise. But, we can use other thresholds to adjust the behavior, allowing the model to be more, or less, eager to predict the positive class. For example, if a threshold of 0.3 is used, the positive class will be predicted if the model predicted the positive class with a probability of 0.3 or higher. So, compared to using 0.5 as the threshold, more predictions of the positive class will be made, increasing both false positives and true positives. 

## Thresholds in multi-class classification
Where we have multiple classes in the target column, if we wish to to set a threshold, it's necessary to also specify one of the classes as the default class. In many cases, this can be fairly natural. For example, if the target column represents medical conditions, the default class may be "No Issues" and the other classes may each relate to specific conditions. Or, if the data represents network logs and the target column relates to intrusion types, then the default may be "Normal Behavior" with the other classes each relating to specific network attacks.

For example, we may have a dataset with four network attacks, with the target column containing the classes: "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For any record for which we run prediction, we will get a probability of each class, which will sum to 1.0. We may get, for example: 0.3, 0.4, 0.1, 0.2 (we assume here this orders the four classes as above). Normally, we would predict "Buffer Overflow" as this has the highest probability, 0.4. However, we can set a threshold in order to modify this behavior, which can affect the rate of false negatives and false positives. 

We may specify, for example: the default class is 'Normal Behavior", the threshold for "Buffer Overflow" is 0.5, for "Port Scan" is 0.55 and for "Phishing" is 0.45. 

Then to make a prediction for any given record, we consider only the classes where the probability is over the specified threshold. In this example, none of the probabilites are over the thresholds, so the default class, "Normal Behavior" is predicted.

If the predicted probabilities were: 0.1, 0.6, 0.2, 0.1, then we would predict "Buffer Overflow": the probability (0.6) is over the threshold (0.5) and this is the highest prediction.

If the predicted probabilites were: 0.1, 0.3, 0.6, 0.0, then we would predict "Port Scan". 

If two or more classes have predicted probabilities over their threshold, we take the higher. If none do, we take the default class.

If the default class has the highest predicted probability, it will be predicted. 

## AUROC and F1 Scores
Numerous metrics are uses for classification problems, but for simplicity, we will consider for the moment just two of the more common, the Area Under a Receiver Operator Curver (AUROC) and the F1 score (specifially, the macro score). These are often both useful, though measure two different things. AUROC measures how well-ordered the predictions are. It applies only to binary prediction, but in a multi-class probablem, we can calculate the AUROC score for each class by treating the problem as a one-vs-all problem. For example, we can calculate the AUROC for each of "Normal Behavior", "Buffer Overflow", "Port Scan", and "Phishing". For the AUROC for "Normal Behavior", we treat the problem as predicted "Normal Behavior" vs not "Normal Behavior", and so on.

In a binary classification problem, AUROC evaluates how well the model tends to give higher probability for the positive class to records of the positive class. That is, it looks at the rank order of the probabilities (not the actual probabilities, and not the class predictions). 

F1 Score, on the other hand, looks at the class predictions, not considering the probabilities behind them. Similarly for precision, recall, MCC (Mathew's Correlation Coeficient) and several other metrics. 

Other metrics, such as Brier Score and Log-Loss look at the probabilities themselves. These may be the most relevant, though to optimize these, we often first work to ensure the probabilities are well-ranked (the AUROC is optimized), and then calibrate the model in post-processing, though tuning the model to produce accurate probabilities is also common. 

To create class predictions from probabilities, it's necessary only that the rank order of the probabilites is good. If the AUROC is high, it will be possible to create a set of class predictions with a high F1 score. 

However, it's also possible to have a high AUROC and low F1 Score, or a high F1 Score and low AUROC. The former is likely due to a poor choice of threshold. This can occur, for example, where the data is imbalanced and the AUROC curve hugs the y-axis. This can create an asymetric curve, where the optimal F1 score (or other such metrics) are found using threhsolds other than 0.5. 

The latter is likely due to a particuarly good choice of threhsold (where most thresholds would perform poorly).

AUROC averages over all possible thresholds. In practice, only use 1. AUROC still useful metric, but can be misleading
if use a sub-optimal threshold.  

These are for metrics that check all probabilities. It may be more relevant to look at the top k predictions, to look at
the lift etc, depending on the project.

AUROC is more straightforward when it has a standard, symetric shape, but this is not always the case. 

AUPRC curve also a common metric and also useful. 

## Metrics based on labels
All metrics derived from confusion matrix.
If are probabilities, these are based on the threshold (so each record has a single class preiction, which may be correct or not).

## APIs
The tool provides five APIs: 
- two APIs (print_stats_labels() and print_stats_proba()) to assess the quality of the predictions given a set of ground truth values and predictions (either probabilities or class predictions). 
- two APIs (plot_by_threshold() and describe_slices()) to visualize the implications of using different threhsolds
- one API (tune_threshold()) to optimize the threshold(s) for a specfied metric

If have AUROC of, say, 0.91, may be good. But, just means random positive sample 91% change ranked higher than random
negative sample. If are a small minority, can be clumped together, or spread more evenly. Knowing helps set threshold.

methods specify target_labels to ensure handle in a sensible order. Especially important with binary classification,
so knows which the probabilities refer to where only one set of probs are provided (can provide a pair of probs
per row if want -- and for multiclass, need to).

### print_stats_labels()
This assumes the labels have been selected already. Where only the probabilities are available, this method cannot be called directly, but will be called by print_stats_proba() if that is called.

Display basic metrics related to the predictions. Displays some of the most common metrics: precision, recall and F1 scores. These are shown per class, as well as their macro average. 

This is method is called by print_stats_proba(), but can be called directly if the labels have been set elsewhere.

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




