
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score, log_loss, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.style.use('ggplot')


def performance_metrics(clf, x_test, y_test, target_names):
    """
    Takes a trained sklearn classifier and return key performance metrics 
    (Warning: handles binary targets only and positive label assumed to be 1) 
    :param clf: sklearn classifier
    :param x_test: array of features
    :param y_test: array of actual target values
    :param target_names: list of labels for target in order [0,1] e.g. ['neg', 'pos']
    :return: performance metrics for the clf in a Pandas DataFrame
    """

    y_pred = clf.predict(x_test)
    y_proba = clf.predict_proba(x_test)
    y_prob = clf.predict_proba(x_test).transpose()[1]

    clf_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred, normalize=True)
    roc_auc_macro = roc_auc_score(y_true=y_test, y_score=y_prob, average='macro')
    roc_auc_weighted = roc_auc_score(y_true=y_test, y_score=y_prob, average='weighted')
    gini_coefficient = (2*roc_auc_macro) - 1
    f1_macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
    f1_weighted = f1_score(y_true=y_test, y_pred=y_pred, average='weighted')
    recall_pos = recall_score(y_true=y_test, y_pred=y_pred, pos_label=1, average='binary')
    precision_pos = precision_score(y_true=y_test, y_pred=y_pred, pos_label=1, average='binary')
    log_loss_score = log_loss(y_true=y_test, y_pred=y_proba, eps=1e-15, normalize=True)

    metrics_df = pd.Series({'accuracy': clf_accuracy,
                            'roc_auc': roc_auc_macro,
                            'roc_auc_weighted': roc_auc_weighted,
                            'gini_coefficient': gini_coefficient,
                            'f1': f1_macro,
                            'f1_weighted': f1_weighted,
                            'recall_pos_class': recall_pos,
                            'precision_pos_class': precision_pos,
                            'log_loss': log_loss_score})

    print('\nKey Metrics:')
    print(metrics_df.round(decimals=3))

    print('\nClassification Report:')
    print(classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names))

    return metrics_df


def calculate_lift(clf, x_test, y_test, bins=10):
    """
    Takes a trained sklearn classifier and calculates the lift in each probability bin
    (Warning: handles binary targets only and positive label assumed to be 1)
    :param clf: sklearn classifier
    :param x_test: array of features
    :param y_actual: array of actual target values
    :param bins: The number of bins to create (default=10) - observations will be equally divided
    :return: Pandas DataFrame with key lift related information for each bin
    """

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)

    cols = ['actual_y', 'prob_positive', 'y_pred']
    data = [y_test, y_prob[:, 1], y_pred]
    df = pd.DataFrame(dict(zip(cols, data)))

    df['probability_bins'] = pd.qcut(df['prob_positive'], bins)
    bins_df = df.groupby('probability_bins')

    actual_prob_positive = df['actual_y'].sum()/len(df)
    positive_pop_captured = bins_df['actual_y'].sum()/df['actual_y'].sum()
    cum_positive_pop_captured = positive_pop_captured[::-1].cumsum()[::-1]
    random_positive_pop_captured = np.array([0.1] * 10)
    cum_random_positive_pop_captured = random_positive_pop_captured[::-1].cumsum()[::-1]
    lift_positive = bins_df['actual_y'].sum()/bins_df['actual_y'].count()
    relative_lift_positive = (lift_positive/actual_prob_positive)*100

    lift_metrics_df = pd.DataFrame({'proportion_positive': positive_pop_captured,
                                    'cumulative_prop_positive': cum_positive_pop_captured,
                                    'lift': lift_positive,
                                    'relative_lift': relative_lift_positive,
                                    'baseline_prop_positive': actual_prob_positive,
                                    'random_prop_positive': random_positive_pop_captured,
                                    'random_cumulative_prop_positive': cum_random_positive_pop_captured})

    lift_metrics_df = lift_metrics_df.sort_index(ascending=False)

    return lift_metrics_df


def performance_plots(clf, x_test, y_test, target_names):
    """
    Takes a trained sklearn classifier and returns key performance plots
    (Warning: handles binary targets only and positive label assumed to be 1) 
    :param clf: sklearn classifier
    :param x_test: array of features
    :param y_test: array of actual target values
    :param target_names: list of labels for target in order [0,1] e.g. ['neg', 'pos']
    :return: plots the confusion matrix, ROC curve, Recall-Precision Curve, Lift & Gain plots
    """

    y_pred = clf.predict(x_test)
    y_prob = clf.predict_proba(x_test)
    y = pd.get_dummies(y_test).values

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 20), sharey=False, sharex=False)

    # ax1 = Confusion matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    ax1.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax1.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=12)

    ax1.set_xlabel('Prediction (prob > 0.5)', fontsize=12)
    ax1.set_ylabel('Actual Outcome', fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(target_names, fontsize=12)
    ax1.set_yticklabels(target_names, fontsize=12)
    ax1.grid(False)

    # ax2 = Precision Recall Curve by Target

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(y[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_prob[:, i])

    precision['micro'], recall['micro'], _ = precision_recall_curve(y.ravel(), y_prob.ravel())
    average_precision['micro'] = average_precision_score(y, y_prob, average='micro')

    ax2.plot(recall['micro'], precision['micro'], color='#e5ae38', lw=2,
             label='micro-avg precision-recall (area={:0.2f})'.format(average_precision['micro']))
    ax2.plot(recall[0], precision[0], color='#fc4f30', lw=2,
             label='precision-recall for {} (area={:.2f})'.format(target_names[0], average_precision[0]))
    ax2.plot(recall[1], precision[1], color='#30a2da', lw=2,
             label='precision-recall for {} (area={:.2f})'.format(target_names[1], average_precision[1]))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve by Target')
    ax2.legend(loc=8, frameon=True, fontsize=8)

    # ax3 =  ROC Curve
    fpr, tpr, thresh = roc_curve(y_true=y_test, y_score=y_prob[:, 1], pos_label=1)
    ax3.plot(fpr, tpr, linewidth=2, color='Navy')
    ax3.set_title('ROC Curve')
    ax3.set_xlabel('False Positive Rate (1-Specificity)')
    ax3.set_ylabel('True Positive Rate (Sensitivity)')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # ax4 = FPR, TPR by Probability Threshold
    ax4.plot(thresh, fpr, label='FPR', linewidth=2, color='#fc4f30')
    ax4.plot(thresh, tpr, label='TPR', linewidth=2, color='#30a2da')
    ax4.set_title('False/True Positive Rate by Probability Threshold')
    ax4.set_xlabel('Probability Threshold')
    ax4.set_ylabel('Score')
    ax4.invert_xaxis()
    ax4.set_xlim(1, 0)
    ax4.set_ylim(0, 1)
    ax4.legend(loc=9, frameon=True, fontsize=8, ncol=2)

    # ax5 = Lift
    lift_metrics_df = calculate_lift(clf, x_test, y_test, bins=10)
    l1 = mlines.Line2D([], [], color='#30a2da', marker='None', label='Random')
    l2 = mlines.Line2D([], [], color='#fc4f30', marker='None', label='Model')
    nc = list(range(90, -10, -10))

    ax5.plot(nc, lift_metrics_df['lift'], linewidth=2, color='#fc4f30')
    ax5.plot(nc, lift_metrics_df['baseline_prop_positive'], linewidth=2, color='#30a2da')
    ax5.set_title('% Correct Positive Predictions by Probability Bin')
    ax5.set_xticks(nc)
    ax5.set_xlabel('Probability of Positive Class (binned)')
    ax5.set_ylabel('% Correct Positive')
    ax5.invert_xaxis()
    ax5.set_xlim(90, 0)
    ax5.set_ylim(0, 1)
    ax5.legend(handles=[l1, l2], loc=9, frameon=True, fontsize=8, ncol=2)

    # ax6 = Gain
    nc2 = list(range(10, 110, 10))
    ax6.plot(nc2, lift_metrics_df['cumulative_prop_positive'], linewidth=2, color='#fc4f30')
    ax6.plot(nc2, lift_metrics_df['random_cumulative_prop_positive'], linewidth=2, color='#30a2da')
    ax6.set_title('Cumulative Positive Class Captured')
    ax6.set_xticks(nc2)
    ax6.set_xlabel('% of population')
    ax6.set_ylabel('% positive class')
    ax6.invert_xaxis()
    ax6.set_xlim(10, 100)
    ax6.set_ylim(0, 1)
    ax6.legend(handles=[l1, l2], loc=9, frameon=True, fontsize=8, ncol=2)

    plt.show()

    return fig
