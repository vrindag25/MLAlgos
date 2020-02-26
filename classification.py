def performance_report(y_test,pred_proba, cutoff, flag= True):
    if np.ndim(pred_proba) == 2:
        pred_proba = pred_proba[:,1]
    else:
        pass
    
    auc_roc1 = roc_auc_score(y_test,np.array(pred_proba > cutoff))
    conf_mat = confusion_matrix(y_test,np.array(pred_proba > cutoff))
    report = classification_report(y_test, pred_proba > cutoff)
    return auc_roc1,conf_mat,report

precision, recall, thresholds = precision_recall_curve(y_test, xgb_pred_prob[:,1])

############ Precision Recall Curve ############
# # calculate precision-recall AUC
auc_roc1 = roc_auc_score(y_test,
              np.array(xgb_pred_prob[:,1] > 0.3))

print('AUC ROC', auc_roc1)

# plot the precision-recall curve for the model
plt.plot(thresholds,precision[:-1],  color = 'g', 
         label = 'Precision')
plt.plot( thresholds, recall[:-1],linestyle = '--', color = 'b', label = 'Recall')
plt.xlabel("Threshold")
plt.legend(loc = 'lower left')
