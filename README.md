# SplitPointViz

SplitPointViz is a tool to visualize and to help set the thresholds used for clasificiation problems. Where models produce probabilities for each class, depending on distribution of probabilities and the metric we wish to optimize for, we may acheive better results using different thresholds. The tools automates selecting a threshold and helps understand the choices related to the thresholds selected.

We assume the use of classifiers that produce probabilities for each record and assume that we wish to convert these into concrete predictions, so that for each record we have a prediction of a single class. Normally we would simply predict the class that recieved the highest probability. In the binary classification case, this is equivalant to taking the class that received a predicted probability over 0.5 (though in very rare cases, both classes may recieve a probability of exactly 0.5 for some rows). 

It may be preferable to not follow the default behavior and to set a different threshold, in order to optimize certain metrics or to treat different classes differently. It may be that certain types of errors are more signficant than others, and it is likely the case that the probabilities produced by the models are not well-calibrated (possibly even after specifically calibrating them in a post-processing step).

The tool may be used for both binary classification and multi-class classification, though is simpler in the binary case. With binary probablems, only a single threshold must be found. With multi-class classification, the tool seeks to identify the optimal threshold for each class. As the two cases are somewhat different, we discuss them separately next.

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

F1 Score, on the other hand, looks at the class predictions, not considering the probabilities behind them. 

Other metrics, such as Brier Score and Log-Loss look at the probabilities themselves. But, to create class predictions from probabilities, it's necessary only that the rank order of the probabilites is good. If the AUROC is high, it will be possible to create a set of class predictions with a high F1 score. 

However, it's also possible to have a high AUROC and low F1 Score, or a high F1 Score and low AUROC. The former is likely due to a poor choice of threshold. The latter is likely due to a particuarly good choice of threhsold (where most thresholds would perform poorly).

## APIs
The tool provides five APIs: 
- two to assess the quality of the predictions given a set of ground truth values and predictions (either probabilities or class predictions).
- two to visualize the implications of using different threhsolds
- one to optimize the threshold(s) for a specfied metric






