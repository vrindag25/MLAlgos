# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
# create the PCA instance
sklearn_pca = sklearnPCA(n_components=5)

# fit on data
sklearn_pca.fit(data(var_list])

# transform data
sklearn_transf = sklearn_pca.fit_transform(data[var_list])

# Loadings ( http://www.nxn.se/valent/loadings-with-scikit-learn-pca)
loadings_df = pd.DataFrame(sklearn_pca.components_.T)
loadings_df.index = X.columns
loadings_df.reset_index(inplace = True)
loadings_df

# Take absolute values
loadings_df.iloc[:,1:] = loadings_df.iloc[:,1:].apply(lambda x:abs(x), axis = 0)
#To find important features based on loadings
imp_feat = list(loadings_df.sort_values(by = 0, ascending = False)['index'][:40])
imp_feat = imp_feat + list(loadings_df.sort_values(by = 1, ascending = False)['index'][:40])
imp_feat = imp_feat + list(loadings_df.sort_values(by = 2, ascending = False)['index'][:40])
imp_feat = imp_feat + list(loadings_df.sort_values(by = 3, ascending = False)['index'][:20])
imp_feat = imp_feat + list(loadings_df.sort_values(by = 4, ascending = False)['index'][:20])
imp_feat = list(dict.fromkeys(imp_feat))

for feat in imp_feat:
    print(feat, end = "','") 
