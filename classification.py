##########################Under sampling ###################################
def undersample_train_test_split(X_train, y_train, method, randnum):
    
    for col in X:
        data[col] = data[col].replace(np.nan,0)
    #Method 1
    if method == 'ClusterCentroids':
        
        cc = ClusterCentroids(random_state=randnum,
                              ratio={0: 150})
        X_cc, y_cc = cc.fit_sample(X_train, y_train)
    
        X_cc = pd.DataFrame(X_cc)
        X_cc.columns= x_var
        y_cc = pd.Series(y_cc)
        return(X_cc, y_cc)
        
    elif method == 'RandomUnderSampler':
        #Method 2
        rs = RandomUnderSampler(sampling_strategy=0.8,
                                random_state = randnum)

        X_rs, y_rs = rs.fit_sample(X_train, y_train)
        
        X_rs = pd.DataFrame(X_rs)
        X_rs.columns= x_var
        y_rs = pd.Series(y_rs)
        return(X_rs,y_rs)
    
    elif method == 'default':
        return(X_train, y_train)
    
#################### Light GBM ##############################################
lightgb_model = LGBMClassifier()
# A parameter grid for LGBMClassifier
params = {'max_depth': [ 5, 6, 7],'subsample': [0.7,0.8],'colsample_bytree': [1]}

folds = 5
param_comb = 200

skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

random_search = RandomizedSearchCV(lightgb_model,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='f1',
                                   n_jobs=4,
                                   cv=skf.split(X_train, y_train),
                                   verbose=0,
                                   random_state=1001)

random_search.fit(X, y)
lightgb_model = random_search.best_estimator_
lightgb_fit = lightgb_model.fit(X, y)

##################### Feature Importance ###################################
importance = list(model.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X_train.columns, importance)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
feature_importances = pd.DataFrame(feature_importances, 
             columns = ['Feature' , 'Imp_Score']) 

print(feature_importances)


##################### SHAP ##########################
explainer = shap.TreeExplainer(lightgb_fit)
shap_values = explainer.shap_values(X)
# summarize the effects of all the features
shap_sum = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
importance_df.columns = ['column_name', 'shap_importance']
importance_df = importance_df.sort_values('shap_importance', ascending=False)
importance_df['relative_imp'] = importance_df['shap_importance']/np.sum(shap_sum)
importance_df

def performance_report(y_test,pred_proba, cutoff, flag= True):
    if np.ndim(pred_proba) == 2:
        pred_proba = pred_proba[:,1]
    else:
        pass
    
    auc_roc1 = roc_auc_score(y_test,np.array(pred_proba > cutoff))
    conf_mat = confusion_matrix(y_test,np.array(pred_proba > cutoff))
    report = classification_report(y_test, pred_proba > cutoff)
    return auc_roc1,conf_mat,report

precision, recall, thresholds = precision_recall_curve(y_test, pred_proba[:,1])

############ Precision Recall Curve ############
# # calculate precision-recall AUC
auc_roc = roc_auc_score(y_test,
              np.array(pred_proba[:,1] > 0.3))

print('AUC ROC', auc_roc)

# plot the precision-recall curve for the model
plt.plot(thresholds,precision[:-1],  color = 'g', 
         label = 'Precision')
plt.plot( thresholds, recall[:-1],linestyle = '--', color = 'b', label = 'Recall')
plt.xlabel("Threshold")
plt.legend(loc = 'lower left')

################################ Metric ########################################
def accuracy(confusion_matrix):
    return (model_confusion_matrix[0,0]+model_confusion_matrix[1,1])/model_confusion_matrix.sum()

def precision(confusion_matrix):
    return model_confusion_matrix[1,1]/(model_confusion_matrix[0,1]+model_confusion_matrix[1,1])
    
def recall(confusion_matrix):
    return model_confusion_matrix[1,1]/(model_confusion_matrix[1,0]+model_confusion_matrix[1,1])

def f1_score(confusion_matrix):
    pr = precision(confusion_matrix)
    rc = recall(confusion_matrix)
    return 2*((pr*rc)/(pr+rc))

#To display image
display(Image(os.getcwd() +'\\image.jpg', width=400))
