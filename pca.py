# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
# create the PCA instance
sklearn_pca = sklearnPCA(n_components=5)

# fit on data
sklearn_pca.fit(data(var_list])

# transform data
sklearn_transf = sklearn_pca.fit_transform(data[var_list])

# Loadings ( http://www.nxn.se/valent/loadings-with-scikit-learn-pca)
loadings = pd.DataFrame(sklearn_pca.components_)
loadings_T = loadings.T

loadings_T['Feature']= x_var
loadings_T.columns = ['PC1','PC2','PC3','PC4','PC5','Feature']
loadings_T.sort_values(['PC1','PC2', 'PC3', 'PC4'], ascending = False)
