import os 
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from network_collection import extract_network_data, network_critical_point, group_critical_point
import scipy
import networkx as nx
from collections import Counter
import itertools 
from mutual_framework import network_generate, betaspace
from kcore_KNN_deg import group_index_from_feature_Kmeans, feature_from_network_topology
from kcore_KNN_degree_partition import mutual_multi, PPI_multi, BDP_multi, SIS_multi, CW_multi, genereg_multi, reducednet_effstate, neighborshell_given_core

from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV 
#from xgboost import XGBClassifier, XGBRegressor

fs = 22
ticksize = 20
legendsize= 16
alpha = 0.8
lw = 3
marksize = 8


def save_sparse_A(dynamics, network_type, N, d_list, seed_list):
    """TODO: Docstring for convert_matrix_to_sparse.

    :arg1: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/'
    des_file = os.listdir(des)
    for d in d_list:
        for seed in seed_list:
            save_file = des + f'N={N}_d={d}_seed={seed}_A.npz'
            if not os.path.exists(save_file):
                A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
                scipy.sparse.save_npz(save_file, scipy.sparse.csr_matrix(A) )

    return None

def reduced_network(dynamics, network_type, space, N, d, m, seed, method, tradeoff_para=0.5):
    """TODO: Docstring for reduced_network.

    :dynamics: TODO
    :network_type: TODO
    :m: TODO
    :returns: TODO

    """
    des_A = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/A_matrix/'
    des_file = des_A + f'N={N}_d={d}_seed={seed}_A.npz'
    A = scipy.sparse.load_npz(des_file).toarray()
    G = nx.from_numpy_array(A)
    N_actual = len(A)
    feature = feature_from_network_topology(A, G, space, tradeoff_para, method)
    group_index = group_index_from_feature_Kmeans(feature, m)
    A_reduction_deg_part, net_arguments_reduction_deg_part, _ = reducednet_effstate(A, np.zeros(N_actual), group_index)

    return A_reduction_deg_part

def feature_engineering(dynamics, network_type, N, d_list, seed_list, m, space, method, tradeoff_para, critical_type, threshold_value, survival_threshold, save_feature):
    """TODO: Docstring for feature_engineering.

    :dynamics: TODO
    :network_type: TODO
    :N: TODO
    :d_list: TODO
    :seed_list: TODO
    :returns: TODO

    """
    X = dict()
    wc = dict()

    for d in d_list:
        for seed in seed_list:
            y_reduction, critical_weight = group_critical_point(dynamics, network_type, N, seed, d, m, space, tradeoff_para, method, critical_type, threshold_value, survival_threshold)
            if critical_weight != None:
                wc[(N, tuple(d), tuple(seed))] = critical_weight
                A_reduced = reduced_network(dynamics, network_type, space, N, d, m, seed, method, tradeoff_para=0.5)
                # X[(N, tuple(d), tuple(seed))] = A_reduced.flatten()
                X[(N, tuple(d), tuple(seed))] = np.hstack((np.sum(A_reduced, 0) )) 
    X = np.vstack(list(X.values()))
    y = np.array(list(wc.values()))
    if save_feature:
        des_feature = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ML_Xy/'
        file_X = des_feature + f'X.csv'
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y.reshape(len(y), 1))
        df_X.to_csv(des_feature + f'X_m={m}.csv', index=None, header=None, mode='a')
        df_y.to_csv(des_feature + f'y_m={m}.csv', index=None, header=None, mode='a')
    return X, y

def read_feature_target(dynamics, network_type, m, save_feature):
    """TODO: Docstring for read_feature_target.

    :arg1: TODO
    :returns: TODO

    """
    if save_feature:
        des_feature = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/ML_Xy/'
        X = np.array(pd.read_csv(des_feature + f'X_m={m}.csv'))
        y = np.array(pd.read_csv(des_feature + f'y_m={m}.csv')).squeeze()
    else:
        X, y = feature_engineering(dynamics, network_type, N, d_list, seed_list, m, space, method, tradeoff_para, critical_type, threshold_value, survival_threshold, save_feature)
        y = y.squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def ridge_lasso_poly(X_train, X_test, y_train, y_test, degree_list, alpha_list, l1_l2):
    """TODO: Docstring for baseline_model.

    :X: TODO
    :y: TODO
    :returns: TODO

    """
    rmse_test = []
    rmse_train = []
    df = pd.DataFrame()
    for degree in degree_list:
        for alpha in alpha_list:
            df_i = pd.DataFrame()
            if l1_l2 == 'ridge':
                model = Pipeline([('poly', PolynomialFeatures(degree=degree)), (l1_l2, Ridge(alpha=alpha))])
            else:
                model = Pipeline([('poly', PolynomialFeatures(degree=degree)), (l1_l2, Lasso(alpha=alpha))])
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            rmse_test.append( mean_squared_error(y_test, y_test_pred, squared=False))
            rmse_train.append( mean_squared_error(y_train, y_train_pred, squared=False))
            baseline = np.array([np.min(np.hstack((y_train, y_train_pred))*0.8 ) , np.max(np.hstack((y_train, y_train_pred))*1.2 ) ])
            df_i['train_test'] = ['training'] * len(y_train) + ['test'] * len(y_test)
            df_i['ytrue'] = np.hstack((y_train, y_test))
            df_i['ypred'] = np.hstack((y_train_pred, y_test_pred))
            df_i['degree'] = degree
            df_i['alpha'] = alpha
            df = pd.concat([df, df_i], ignore_index=True)

    g = sns.FacetGrid(df, col='alpha', row='degree', hue='train_test', sharex=True, sharey=True)
    g.map(sns.scatterplot, 'ytrue', 'ypred', alpha=0.3)
    colors = sns.color_palette().as_hex()
    axes = g.fig.axes
    for ax, rmse_train_i, rmse_test_i in zip(axes, rmse_train, rmse_test):
        ax.plot(baseline, baseline, '--k', alpha=0.3)
        ax.text(0.22, 0.08, 'rmse_train:{:g}'.format(float('{:.2g}'.format(rmse_train_i))), color=colors[0])
        ax.text(0.22, 0.04, 'rmse_test:{:g}'.format(float('{:.2g}'.format(rmse_test_i))), color=colors[1])
    g.set_xlabels('$y_{true}$', size=15)
    g.set_ylabels('$y_{pred}$', size=15)

    plt.legend(fontsize=13, frameon=False, markerscale=1.5) 
    plt.locator_params(nbins=5)
    plt.show()

    return y_train, y_train_pred, y_test, y_test_pred, rmse_train, rmse_test

def feature_importance(X_train, X_test, y_train, y_test, m ):
    """TODO: Docstring for feature_importance.

    :X_train: TODO
    :X_test: TODO
    :y_train: TODO
    :y_test: TODO
    :model: TODO
    :returns: TODO

    """
    rf = RandomForestRegressor(random_state=1)
    parameters = {'n_estimators':[100, 200, 500], 'max_depth':[2, 5, 10], 'min_samples_split':[2, 5, 10], 'bootstrap':[True, False], 'max_features':['sqrt', 'log2'] }
    #parameters = {'n_estimators':[100], 'max_depth':[2], 'min_samples_split':[2], 'bootstrap':[True, False], 'max_features':['sqrt', 'log2'] }
    grids, grid_best = cross_valid_model(X_train, y_train, rf, parameters)
    y_train_pred = grids.best_estimator_.predict(X_train)
    y_test_pred = grids.best_estimator_.predict(X_test)
    df = pd.DataFrame()
    df['train_test'] = ['training'] * len(y_train) + ['test'] * len(y_test)
    df['ytrue'] = np.hstack((y_train, y_test))
    df['ypred'] = np.hstack((y_train_pred, y_test_pred))
    sns.scatterplot(data=df, x='ytrue', y='ypred', hue='train_test') 
    colors = sns.color_palette().as_hex()

    baseline = np.array([np.min(np.hstack((y_train, y_train_pred))*0.8 ) , np.max(np.hstack((y_train, y_train_pred))*1.2 ) ])
    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    plt.plot(baseline, baseline, '--k', alpha=0.3)

    plt.text(0.22, 0.08, 'rmse_train:{:g}'.format(float('{:.2g}'.format(rmse_train))), color=colors[0], size=16)
    plt.text(0.22, 0.04, 'rmse_test:{:g}'.format(float('{:.2g}'.format(rmse_test))), color=colors[1], size=16)
    plt.xlabel('$y_{true}$', fontsize=20)
    plt.ylabel('$y_{pred}$', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=16, frameon=False, markerscale=1.5) 
    plt.locator_params(nbins=5)
    plt.subplots_adjust(left=0.15, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.savefig(f'../report/report042722/rf_y_true_pred_m={m}.png')
    plt.close()
    
    
    importance = grids.best_estimator_.feature_importances_.reshape(m, m)

    g = sns.heatmap(importance, xticklabels=np.arange(1, m+1, 1), yticklabels=np.arange(1, m+1, 1), square=True, cmap="YlGnBu",  annot=True, fmt='.2f', annot_kws={'fontsize':13})
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.locator_params(nbins=5)
    g.tick_params(labelsize=18)
    plt.savefig(f'../report/report042722/rf_feature_importance_m={m}.png')
    plt.close()
    return None

def sensitivity_connection(network_type, dynamics, seed_list, d, weight_list):
    """TODO: Docstring for sensitivity_connection.

    :network_type: TODO
    :dynamics: TODO
    :seed: TODO
    :d: TODO
    :weight_list: TODO
    :returns: TODO

    """
    des = '../data/' + dynamics + '/' + network_type + '/xs_bifurcation/xs_multi/' 
    for seed in seed_list:
        des_file = des + f'N={N}_d={d}_seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        A_unit, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, seed, d)
        index = [np.where(np.abs(w-data[:, 0]) < 1e-8)[0][0] for w in weight_list]
        xs_multi = data[index, 1:]
        y_multi = betaspace(A_unit, xs_multi)[-1]
        plt.plot(weight_list, y_multi, label=f'seed={seed[1]}')
    plt.legend()
    plt.show()
    return None
 

        




dynamics = 'mutual'
network_type = 'SF'
space = 'log'
tradeoff_para = 0.5
method = 'degree'
critical_type = 'y_gl'
threshold_value = 2
survival_threshold = 5
m = 5

N = 1000
seed_list = [[i, i] for i in range(0, 50)]

gamma_list = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
gamma_list = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
kmin_list = [3, 4, 5]
kmax_list = [N-1]
d_list = [list(i) for i in list(itertools.product(gamma_list, kmax_list, kmin_list))]

m_list = [3]
for  m in m_list:
    #X, y = feature_engineering(dynamics, network_type, N, d_list, seed_list, m, space, method, tradeoff_para, critical_type, threshold_value, survival_threshold, save_feature=0)

    pass





gaussnb = GaussianNB()
linearreg = LinearRegression(fit_intercept=True)
lasso = Lasso()
ridge = Ridge()
baysridge = BayesianRidge()
logisreg = LogisticRegression()
svr = SVR(kernel='linear')
knn = KNeighborsRegressor()
#xgb = XGBRegressor(random_state=0, use_label_encoder=False)
rf = RandomForestRegressor(n_estimators=100, random_state=1)

m = 3
X_train, X_test, y_train, y_test = read_feature_target(dynamics, network_type, m, save_feature=0)
degree_list = [1, 2, 3, 4, 5, 6]
l1_l2 = 'lasso'
alpha_list = [0, 0.0001, 0.001, 0.01, 0.1]
l1_l2 = 'ridge'
alpha_list = [0, 0.01, 0.1, 1, 10]
ridge_lasso_poly(X_train, X_test, y_train, y_test, degree_list, alpha_list, l1_l2)



for m in [3]:
    #X_train, X_test, y_train, y_test = read_feature_target(dynamics, network_type, m)
    #feature_importance(X_train, X_test, y_train, y_test, m )
    pass



