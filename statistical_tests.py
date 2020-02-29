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
    
    i = i+1
