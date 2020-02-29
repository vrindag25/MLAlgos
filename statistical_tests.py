from scipy.stats import kruskal
p_value_table = df_empty(columns = ['feature',
                                        'ttest_pvalue',
                                        'mannwitney_pvalue',
                                        'kruskal_pvalue'],
                             dtypes = [np.object,
                                       np.float32,
                                       np.float32,
                                       np.float32,
                                    ]
                            )
i = 0
for col in data[num_vars]:
    print(col, 'done')
    p_value_table.loc[i, 'feature'] = col
    
    x = train_data.loc[train_data[target] == 0,col].dropna()
    y = train_data.loc[train_data[target] == 1,col].dropna()
#     z = train_data.loc[train_data[target] == 2,col].dropna()
    res1 = mannwhitneyu(x,y)   
    p_value_table.loc[i, 'mannwitney_pvalue'] = round(res1.pvalue,4)
    
    res= sc.stats.ttest_ind(x,y,equal_var = False)
    p_value_table.loc[i, 'ttest_pvalue'] = round(res.pvalue,4)
    
    res2 = kruskal(x,y)
    p_value_table.loc[i, 'kruskal_pvalue'] = round(res2.pvalue,4)
    
    
  #################SKLEARN########################

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x1, y)
# Predict
y_predicted = regression_model.predict(x1)

##############StatsModel#######################
newX = sm.add_constant(x1)
regression_model_sm = sm.OLS(y, newX)
regression_model_sm_fit = regression_model_sm.fit()
print(regression_model_sm_fit.summary())

print("------------------Variation Explained by Coefficients -----------------")
beta_coef = regression_model.coef_
alpha_coef = regression_model.intercept_

print(Fore.GREEN +'Variation Explained by alpha: ',
      alpha_coef[0]/y.mean())

print(Fore.GREEN +'Variation Explained by beta: ',
      beta_coef[0,0]*x1.mean()/y.mean()) # ,beta_coef[0,1]*x2.mean()/y.mean())
# plotting values

# data points
plt.scatter(x1, y) #, s=10
plt.xlabel('Program Utilisation')
plt.ylabel('Reversal Rate')

# predicted values
plt.plot(x1, y_predicted, color='r')
plt.show()
    i = i+1
