def performance_report(y_test,pred_proba, cutoff, flag= True):
    if np.ndim(pred_proba) == 2:
        pred_proba = pred_proba[:,1]
    else:
        pass
    
    auc_roc1 = roc_auc_score(y_test,np.array(pred_proba > cutoff))
    conf_mat = confusion_matrix(y_test,np.array(pred_proba > cutoff))
    report = classification_report(y_test, pred_proba > cutoff)
    return auc_roc1,conf_mat,report
