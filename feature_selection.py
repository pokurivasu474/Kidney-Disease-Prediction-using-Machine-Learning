from sklearn.feature_selection import SelectKBest, mutual_info_classif

def feature_selection(X, y, k=15):
    # Select the top k features based on mutual information
    selector = SelectKBest(mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return selector, X_selected