import pandas as pd
import dask.dataframe as dd
import itertools
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.utils import shuffle
from statsmodels.stats import multitest, anova as aov
import dabest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats
from scipy.special import factorial, betainc
import umap
from sklearn import preprocessing, ensemble, cluster
from scipy import stats
import pingouin as pg
import numpy as np
import networkx as nx
import community
import math
from fancyimpute import KNN
import kmapper as km
from report_manager import utils
from report_manager.analyses import wgcnaAnalysis as wgcna
import statsmodels.api as sm
from statsmodels.formula.api import ols
import time
from joblib import Parallel, delayed
from numba import jit
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri

pandas2ri.activate()

R = ro.r
base = importr('base')
stats_r = importr('stats')
samr = importr('samr')

def transform_into_wide_format(data, index, columns, values, extra=[]):
    df = data.copy()
    if not df.empty:
        cols = [columns, values]
        cols.extend(index)
        if len(extra) > 0:
            extra.extend(index)
            extra_cols = df[extra].set_index(index)
        df = df[cols]
        df = df.pivot_table(index=index, columns=columns, values=values)
        df = df.join(extra_cols)
        df = df.drop_duplicates()
        df = df.reset_index()

    return df

def transform_into_long_format(data, drop_columns, group, columns=['mame','y']):
    data = data.drop(drop_columns, axis=1)
    
    long_data = pd.melt(data, id_vars=group, var_name=columns[0], value_name=columns[1])
    long_data = long_data.set_index(group)
    long_data.columns = columns
    
    return long_data

def get_ranking_with_markers(data, drop_columns, group, columns, list_markers, annotation={}):
    long_data = transform_into_long_format(data, drop_columns, group, columns)
    if len(set(long_data['name'].values.tolist()).intersection(list_markers)) > 0:
        long_data = long_data.drop_duplicates()
        long_data['symbol'] = [ 17 if p in list_markers else 0 for p in long_data['name'].tolist()]
        long_data['size'] = [25 if p in list_markers else 7 for p in long_data['name'].tolist()]
        long_data['name'] = [p+' marker in '+annotation[p] if p in annotation else p for p in long_data['name'].tolist()]
        
    return long_data


def extract_number_missing(df, missing_max, drop_cols=['sample'], group='group'):
    if group is None:
        groups = df.loc[:, df.notnull().sum(axis = 1) >= missing_max]
    else:
        groups = df.copy()
        groups = groups.drop(drop_cols, axis = 1)
        groups = groups.set_index(group).notnull().groupby(level=0).sum(axis = 1)
        groups = groups[groups>=missing_max]

    groups = groups.dropna(how='all', axis=1)
    return list(groups.columns)

def extract_percentage_missing(data, missing_max, drop_cols=['sample'], group='group'):
    if group is None:
        groups = data.loc[:, data.isnull().mean() <= missing_max].columns
    else:
        groups = data.copy()
        groups = groups.drop(drop_cols, axis = 1)
        groups = groups.set_index(group)
        groups = groups.isnull().groupby(level=0).mean()
        groups = groups[groups<=missing_max]
        groups = groups.dropna(how='all', axis=1).columns

    return list(groups)

def imputation_KNN(data, drop_cols=['group', 'sample', 'subject'], group='group', cutoff=0.6, alone = True):
    df = data.copy()
    value_cols = [c for c in df.columns if c not in drop_cols]
    for g in df[group].unique():
        missDf = df.loc[df[group]==g, value_cols]
        missDf = missDf.loc[:, missDf.notnull().mean() >= cutoff]
        if missDf.isnull().values.any():
            X = np.array(missDf.values, dtype=np.float64)
            X_trans = KNN(k=3,verbose=False).fit_transform(X)
            missingdata_df = missDf.columns.tolist()
            dfm = pd.DataFrame(X_trans, index =list(missDf.index), columns = missingdata_df)
            df.update(dfm)
    if alone:
        df = df.dropna()

    return df

def imputation_mixed_norm_KNN(data, index_cols=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3, group='group', cutoff=0.6):
    df = imputation_KNN(data, drop_cols=index_cols, group=group, cutoff=cutoff, alone = False)
    df = imputation_normal_distribution(df, index_cols=index_cols, shift=shift, nstd=nstd)

    return df

def imputation_normal_distribution(data, index_cols=['group', 'sample', 'subject'], shift = 1.8, nstd = 0.3):
    np.random.seed(112736)
    df = data.copy()
    if index_cols is not None:
        df = df.set_index(index_cols)
    data_imputed = df.T.sort_index()
    null_columns = data_imputed.columns[data_imputed.isnull().any()]
    for c in null_columns:
        missing = data_imputed[data_imputed[c].isnull()].index.tolist()
        std = data_imputed[c].std()
        mean = data_imputed[c].mean()
        sigma = std*nstd
        mu = mean - (std*shift)
        value = 0.0
        if not math.isnan(std) and not math.isnan(mean) and not math.isnan(sigma) and not math.isnan(mu):
            value = np.random.normal(mu, sigma, size=len(missing))
        
        data_imputed.loc[missing, c] = value

    return data_imputed.T


def polish_median_normalization(data, max_iter = 10):
    mediandf = data.copy()
    for i in range(max_iter):
        col_median = mediandf.median(axis= 0)
        row_median = mediandf.median(axis = 1)
        if row_median.mean() == 0 and col_median.mean() ==0:
            break

        mediandf = mediandf.sub(row_median, axis=0)
        mediandf = mediandf.sub(col_median, axis=1)

    normData = data - mediandf

    return normData

def quantile_normalization(data):
    rank_mean = data.stack().groupby(data.rank(method='first').stack().astype(int)).mean()
    normdf = data.rank(method='min').stack().astype(int).map(rank_mean).unstack()

    return normdf

def linear_normalization(data, method = "l1", axis = 0):
    normvalues = preprocessing.normalize(data.fillna(0).values, norm=method, axis=axis, copy=True, return_norm=False)
    normdf = pd.DataFrame(normvalues, index = data.index, columns = data.columns)

    return normdf

def remove_group(data):
    data.drop(['group'], axis=1)
    return data

def calculate_coefficient_variation(values):
    cv = scipy.stats.variation(values.apply(lambda x: np.power(2,x)).values) *100
    
    return cv

def get_coefficient_variation(data, drop_columns, group, columns):
    df = data.copy()
    formated_df = df.drop(drop_columns, axis=1)
    cvs = formated_df.groupby(group).apply(func=calculate_coefficient_variation)
    cols = formated_df.set_index(group).columns.tolist()
    cvs_df = pd.DataFrame()
    for i in cvs.index:
        gcvs = cvs[i].tolist()
        ints = formated_df.set_index('group').mean().values.tolist()
        tdf = pd.DataFrame(data= {'name':cols, 'x':gcvs, 'y':ints})
        tdf['group'] = i
        
        if cvs_df.empty:
            cvs_df = tdf.copy()
        else:
            cvs_df = cvs_df.append(tdf)

    return cvs_df


def get_proteomics_measurements_ready(df, index_cols=['group', 'sample', 'subject'], drop_cols=['sample'], group='group', identifier='identifier', extra_identifier='name', imputation = True, method = 'distribution', missing_method = 'percentage', missing_per_group=True, missing_max = 0.3, value_col='LFQ_intensity'):
    df = df.set_index(index_cols)
    if extra_identifier is not None and extra_identifier in df.columns:
        df[identifier] = df[extra_identifier].map(str) + "~" + df[identifier].map(str)
    df = df.pivot_table(values=value_col, index=df.index, columns=identifier, aggfunc='first')
    df = df.reset_index()
    df[index_cols] = df["index"].apply(pd.Series)
    df = df.drop(["index"], axis=1)
    aux = index_cols
    if missing_per_group == False:
        group = None
    if missing_method == 'at_least_x':
        aux.extend(extract_number_missing(df, missing_max, drop_cols, group=group))
    elif missing_method == 'percentage':
        aux.extend(extract_percentage_missing(df,  missing_max, drop_cols, group=group))

    df = df[list(set(aux))]
    if imputation:
        if method == "KNN":
            df = imputation_KNN(df)
        elif method == "distribution":
            df = imputation_normal_distribution(df, shift = 1.8, nstd = 0.3)
        elif method == 'mixed':
            df = imputation_mixed_norm_KNN(df)
            
        df = df.reset_index()

    return df

def get_clinical_measurements_ready(df, subject_id='subject', sample_id='biological_sample', group_id='group', columns=['clinical_variable'], values='values', extra=['group'], imputation=True, imputation_method='KNN'):
    index = [subject_id, sample_id]
    drop_cols = [subject_id, sample_id]
    drop_cols.append(group_id)
    
    processed_df = transform_into_wide_format(df, index=index, columns=columns, values=values, extra=extra)
    if imputation:
        if imputation_method.lower() == "knn":
            df = imputation_KNN(processed_df, drop_cols=drop_cols, group=group_id)
        elif imputation_method.lower() == "distribution":
            df = imputation_normal_distribution(processed_df, index_cols=index)
        elif imputation_method.lower() == 'mixed':
            df = imputation_mixed_norm_KNN(processed_df,index_cols=index, group=group_id)
    
    #df = df.set_index(index)
    
    return df


def run_pca(data, drop_cols=['sample', 'subject'], group='group', components=2, dropna=True):
    np.random.seed(112736)
    result = {}
    args = {}
    df = data.copy()
    if set(drop_cols).intersection(df.columns) == len(drop_cols):
        df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    df = df.select_dtypes(['number'])
    if dropna:
        df = df.dropna(axis=1)
    X = df.values
    y = df.index
    if X.size > 0:
        pca = PCA(n_components=components)
        X = pca.fit_transform(X)
        var_exp = pca.explained_variance_ratio_
        loadings = pd.DataFrame(pca.components_.transpose() * np.sqrt(pca.explained_variance_))
        loadings.index = df.columns
        loadings.columns = ['x', 'y']
        loadings['value'] = np.sqrt(np.power(loadings['x'],2) + np.power(loadings['y'],2))
        loadings = loadings.sort_values(by='value', ascending=False)
        args = {"x_title":"PC1"+" ({0:.2f})".format(var_exp[0]),"y_title":"PC2"+" ({0:.2f})".format(var_exp[1])}
        if components == 2:
            resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
            resultDf = resultDf.reset_index()
            resultDf.columns = ["name", "x", "y"]
        if components > 2:
            args.update({"z_title":"PC3"+str(var_exp[2])})
            resultDf = pd.DataFrame(X, index = y)
            resultDf = resultDf.reset_index()
            cols = []
            if components>3:
                cols = resultDf.columns[4:]
            resultDf.columns = ["name", "x", "y", "z"] + cols

        result['pca'] = (resultDf, loadings)
    
    return result, args

def run_tsne(data, drop_cols=['sample', 'subject'], group='group', components=2, perplexity=40, n_iter=1000, init='pca', dropna=True):
    result = {}
    args = {}
    df = data.copy()
    if set(drop_cols).intersection(df.columns) == len(drop_cols):
        df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    if dropna:
        df = df.dropna(axis=1)
    df = df.select_dtypes(['number'])
    X = df.values
    y = df.index
    if X.size > 0:
        tsne = TSNE(n_components=components, verbose=0, perplexity=perplexity, n_iter=n_iter, init=init)
        X = tsne.fit_transform(X)
        args = {"x_title":"C1","y_title":"C2"}
        if components == 2:
            resultDf = pd.DataFrame(X, index = y, columns = ["x","y"])
            resultDf = resultDf.reset_index()
            resultDf.columns = ["name", "x", "y"]
        if components > 2:
            args.update({"z_title":"C3"})
            resultDf = pd.DataFrame(X, index = y)
            resultDf = resultDf.reset_index()
            cols = []
            if len(components)>4:
                cols = resultDf.columns[4:]
            resultDf.columns = ["name", "x", "y", "z"] + cols
        result['tsne'] = resultDf
    return result, args

def run_umap(data, drop_cols=['sample', 'subject'], group='group', n_neighbors=10, min_dist=0.3, metric='cosine', dropna=True):
    result = {}
    args = {}
    df = data.copy()
    if set(drop_cols).intersection(df.columns) == len(drop_cols):
        df = df.drop(drop_cols, axis=1)
    df = df.set_index(group)
    if dropna:
        df = df.dropna(axis=1)
    df = df.select_dtypes(['number'])
    X = df.values
    y = df.index
    if X.size:
        X = umap.UMAP(n_neighbors=10, min_dist=0.3, metric= metric).fit_transform(X)
        args = {"x_title":"C1","y_title":"C2"}
        resultDf = pd.DataFrame(X, index = y)
        resultDf = resultDf.reset_index()
        cols = []
        if len(resultDf.columns)>3:
                cols = resultDf.columns[3:]
        resultDf.columns = ["name", "x", "y"] + cols
        result['umap'] = resultDf
    return result, args

def calculate_correlations(x, y, method='pearson'):
    if method == "pearson":
        coefficient, pvalue = stats.pearsonr(x, y)
    elif method == "spearman":
        coefficient, pvalue = stats.spearmanr(x, y)

    return (coefficient, pvalue)

def apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep'):
    rejected, padj = multitest.fdrcorrection(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_twostage_fdrcorrection(pvalues, alpha=0.05, method='bh'):
    rejected, padj, num_hyp, alpha_stages = multitest.fdrcorrection_twostage(pvalues, alpha, method)

    return (rejected, padj)

def apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group, alpha=0.05, permutations=50):
    #initial_seed = 176782
    i = permutations
    df_index = df.index.values
    #df_columns = df.columns.values
    seen = [''.join(df_index)]
    #columns = ['identifier']
    rand_pvalues = []
    while i>0:
        df_index = shuffle(df_index)
        #df_columns = shuffle(df_columns)
        df_random = df.reset_index(drop=True)
        df_random.index = df_index
        df_random.index.name = group
        #df_random.columns = df_columns
        #columns = ['identifier', 'F-statistics', 'pvalue_'+str(i)]
        if ''.join(list(df_random.index)) not in seen:
            seen.append(''.join(list(df_random.index)))
            #aov_results = []
            for col in df_random.columns:
                rows = df_random[col]
                #aov_results.append((col,)+calculate_anova(rows, group=group))
                rand_pvalues.append(calculate_anova(rows,group=group)[1])
            #rand_scores = pd.DataFrame(aov_results, columns=columns)
            #rand_scores = rand_scores.set_index("identifier")
            #rand_scores = rand_scores.dropna(how="all")
            #rand_scores = rand_scores[['pvalue_'+str(i)]]
            #rand_pvalues.append(rand_scores)
            i -= 1
    #rand_pvalues = pd.concat(rand_pvalues, axis=1)
    rand_pvalues = np.array(rand_pvalues)
    count = observed_pvalues.to_frame().apply(func=get_counts_permutation_fdr, result_type='expand', axis=1, args=(rand_pvalues, observed_pvalues, permutations, alpha))
    count.columns = ['padj', 'rejected']
    
    return count

def get_counts_permutation_fdr(value, random, observed, n, alpha):
    a = random[random <= value.values[0]].shape[0] + 0.01 #Offset in case of a = 0.0
    b = (observed <= value.values[0]).sum()
    qvalue = (a/b * 1/float(n))
    
    return (qvalue, qvalue <= alpha)

def convertToEdgeList(data, cols):
    data.index.name = None
    edge_list = data.stack().reset_index()
    edge_list.columns = cols

    return edge_list

def run_correlation(df, alpha=0.05, subject='subject', group='group', method='pearson', correction=('fdr', 'indep')):
    correlation = pd.DataFrame()
    if check_is_paired(df, subject, group):
        if len(df[subject].unique()) > 2:
            correlation = run_rm_correlation(df, alpha=alpha, subject=subject, correction=correction)
    else:
        df = df.dropna(axis=1)._get_numeric_data()
        if not df.empty:
            r, p = run_efficient_correlation(df, method=method)
            rdf = pd.DataFrame(r, index=df.columns, columns=df.columns)
            pdf = pd.DataFrame(p, index=df.columns, columns=df.columns)
            correlation = convertToEdgeList(rdf, ["node1", "node2", "weight"])
            pvalues = convertToEdgeList(pdf, ["node1", "node2", "pvalue"])

            if correction[0] == 'fdr':
                rejected, padj = apply_pvalue_fdrcorrection(pvalues["pvalue"].tolist(), alpha=alpha, method=correction[1])
            elif correction[0] == '2fdr':
                rejected, padj = apply_pvalue_twostage_fdrcorrection(pvalues["pvalue"].tolist(), alpha=alpha, method=correction[1])

            correlation["padj"] = padj
            correlation["rejected"] = rejected
            correlation = correlation[correlation.rejected]
    return correlation

def run_multi_correlation(df, alpha=0.05, subject='subject', on=['subject', 'biological_sample'] , group='group', method='pearson', correction=('fdr', 'indep')):
    multidf = pd.DataFrame()
    correlation = None
    if len(df) > 1:
        for dtype in df:
            if multidf.empty:
                multidf = df[dtype]
            else:
                multidf = pd.merge(multidf, df[dtype], how='inner', on=on)
        
        correlation = run_correlation(multidf, alpha=0.05, subject=subject, group=group, method=method, correction=correction)
    
    return correlation
    
def calculate_rm_correlation(df, x, y, subject):
    # ANCOVA model
    cols = ["col0","col1", subject]
    a = "col0"
    b = "col1"
    df.columns = cols

    formula = b + ' ~ ' + 'C(' + subject + ') + ' + a
    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=3)
    # Extract the sign of the correlation and dof
    sign = np.sign(model.params[a])
    dof = int(table.loc['Residual', 'df'])
    # Extract correlation coefficient from sum of squares
    ssfactor = table.loc[a, 'sum_sq']
    sserror = table.loc['Residual', 'sum_sq']
    rm = sign * np.sqrt(ssfactor / (ssfactor + sserror))
    # Extract p-value
    pvalue = table.loc[a, 'PR(>F)']
    pvalue *= 0.5
    
    #r, dof, pvalue, ci, power = pg.rm_corr(data=df, x=x, y=y, subject=subject)
    
    return (x, y, rm, pvalue, dof)

def run_rm_correlation(df, alpha=0.05, subject='subject', correction=('fdr', 'indep')):
    rows = []
    if not df.empty:
        df = df.set_index(subject)._get_numeric_data().dropna(axis=1)
        combinations = itertools.combinations(df.columns, 2)
        df = df.reset_index()
        for x, y in combinations:
            subset = df[[x,y, subject]]
            row = calculate_rm_correlation(subset, x, y, subject)
            rows.append(row)
        correlation = pd.DataFrame(rows, columns=["node1", "node2", "weight", "pvalue", "dof"])
        
        if correction[0] == 'fdr':
            rejected, padj = apply_pvalue_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])
        elif correction[0] == '2fdr':
            rejected, padj = apply_pvalue_twostage_fdrcorrection(correlation["pvalue"].tolist(), alpha=alpha, method=correction[1])

        correlation["padj"] = padj
        correlation["rejected"] = rejected
        correlation = correlation[correlation.rejected]
    return correlation

def run_efficient_correlation(data, method='pearson'):
    matrix = data.values
    if method == 'pearson':
        r = np.corrcoef(matrix, rowvar=False)
    elif method == 'spearman':
        r, p = stats.spearmanr(matrix, axis=0)

    diagonal = np.triu_indices(r.shape[0],1)
    rf = r[diagonal]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = pf
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    
    r[diagonal] = np.nan 
    p[diagonal] = np.nan

    return r, p

def calculate_paired_ttest(df, condition1, condition2):
    group1 = df[[condition1]].values
    group2 = df[[condition2]].values
    
    mean1 = group1.mean() 
    mean2 = group2.mean()
    log2fc = mean1 - mean2
    t, pvalue = stats.ttest_rel(group1, group2, nan_policy='omit')

    return (t, pvalue, mean1, mean2, log2fc)

def calculate_ttest_samr(df, labels, n=2, s0=0, paired=False):
    conditions = df.columns.unique()
    mean1 = df[conditions[0]].mean(axis=1)
    mean2 = df[conditions[1]].mean(axis=1)
    # log2fc = mean1 - mean2

    if paired:
        counts = {}
        labels = []
        for col in df.columns:
            cur_count = counts.get(col, 0)
            labels.append([(cur_count+1)*-1 if col == conditions[0] else (cur_count+1)][0])
            counts[col] = cur_count + 1

        ttest_res = samr.paired_ttest_func(df.values, base.unlist(labels), s0=s0)
    else:
        ttest_res = samr.ttest_func(df.values, base.unlist(labels), s0=s0)
    
    pvalues = [2*stats_r.pt(-base.abs(i), df=n-1)[0] for i in ttest_res[0]]

    result = pd.DataFrame([df.index, mean1, mean2, ttest_res[1], ttest_res[0], pvalues]).T
    result.columns = ['identifier', 'mean(group1)', 'mean(group2)', 'log2FC', 't-statistics', 'pvalue']
    result['group1'] = conditions[0]
    result['group2'] = conditions[1]
    result['FC'] = [np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)) for x in result['log2FC'].values]
    result = result[['identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'FC', 't-statistics', 'pvalue']]

    return result

def calculate_ttest(df, condition1, condition2):
    group1 = df[[condition1]].values
    group2 = df[[condition2]].values
    
    mean1 = group1.mean() 
    mean2 = group2.mean()
    log2fc = mean1 - mean2
    t, pvalue = stats.ttest_ind(group1, group2, nan_policy='omit')

    return (t, pvalue, mean1, mean2, log2fc)

def calculate_THSD(df, group='group', alpha=0.05):
    col = df.name
    df_results = pg.pairwise_tukey(dv=col, between=group, data=pd.DataFrame(df).reset_index(), alpha=alpha, tail='two-sided')
    df_results.columns = ['group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'std_error', 'tail', 't-statistics', 'padj_THSD', 'effsize', 'efftype']
    df_results['identifier'] = col
    df_results = df_results.set_index('identifier')
    df_results['FC'] = df_results['log2FC'].apply(lambda x: np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)))
    df_results['rejected'] = df_results['padj_THSD'].apply(lambda x: True if x < alpha else False)

    return df_results

def calculate_pairwise_ttest(df, column, subject='subject', group='group', correction='none'):
    posthoc_columns = ['Contrast', 'group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired', 'Parametric', 'T', 'dof', 'tail', 'padj', 'BF10', 'efsize', 'eftype']
    if correction == "none":
        valid_cols = ['group1', 'group2', 'mean(group1)', 'std(group1)', 'mean(group2)', 'std(group2)', 'Paired','Parametric', 'T', 'dof', 'BF10', 'efsize', 'eftype']
    else:
        valid_cols = posthoc_columns
    posthoc = pg.pairwise_ttests(data=df, dv=column, between=group, subject=subject, effsize='hedges', return_desc=True, padjust=correction)
    posthoc.columns =  posthoc_columns
    posthoc = posthoc[valid_cols]
    posthoc = complement_posthoc(posthoc, column)
    posthoc = posthoc.set_index('identifier')

    return posthoc

def complement_posthoc(posthoc, identifier):
    posthoc['identifier'] = identifier
    posthoc['log2FC'] = posthoc['mean(group1)'] -posthoc['mean(group2)']
    posthoc['FC'] = posthoc['log2FC'].apply(lambda x: np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)))

    return posthoc

def calculate_dabest(df, idx, x, y, paired=False, id_col=None, test='mean_diff'):
    cols = ["group1", "group2", "effect size", "paired", 'difference', "CI", "bca low", "bca high", "bca interval idx", "pct low", "pct high", "pct interval idx", "bootstraps", 'resamples', 'random seed', 'pvalue Welch', 'statistic Welch', 'pvalue Student T', 'statistic Student T', 'pvalue Mann Whitney', 'statistic Mann Whitney']
    valid_cols = ["group1", "group2", "effect size", "paired", 'difference', "CI", 'pvalue Welch', 'statistic Welch', 'pvalue Student T', 'statistic Student T', 'pvalue Mann Whitney', 'statistic Mann Whitney']
    dabest_df = dabest.load(df, idx=idx, x=x, y=y, paired=paired, id_col=id_col)
    result = pd.DataFrame()
    if test == 'mean_diff':
        result = dabest_df.mean_diff.results
    elif test == 'median_diff':
        result = dabest_df.median_diff.results
    elif test == 'cohens_d':
        result = dabest_df.cohens_d.results
    elif test == 'hedges_g':
        result = dabest_df.hedges_g.results
    elif test == 'cliffs_delta':
        result = dabest_df.cliffs_delta
    
    result.columns = cols
    result = result[valid_cols]

    result['identifier'] = y

    return result

def calculate_anova_samr(df, labels, n=2, s0=0):
    aov_res = samr.multiclass_func(df.values, base.unlist(labels), s0=s0)
    pvalues = [2*stats_r.pt(-base.abs(i), df=n-1)[0] for i in aov_res[0]]
    
    result = pd.DataFrame([df.index, aov_res[0], pvalues]).T
    result.columns = ['identifier', 'F-statistics', 'pvalue']    

    return result

def calculate_anova(df, group='group'):
    group_values = df.groupby(group).apply(np.array).values
    t, pvalue = stats.f_oneway(*group_values)
    
    return (t, pvalue)

def calculate_repeated_measures_anova(df, column, subject='subject', group='group', alpha=0.05):
    aov_result = pg.rm_anova(data=df, dv=column, within=group,subject=subject, detailed=True, correction=True)
    aov_result.columns = ['Source', 'SS', 'DF', 'MS', 'F', 'pvalue', 'padj', 'np2', 'eps', 'sphericity', 'Mauchlys sphericity', 'p-spher']
    t, pvalue = aov_result.loc[0, ['F', 'pvalue']].values 

    return (column, t, pvalue)

def get_max_permutations(df, group='group'):
    num_groups = len(list(df.index))
    num_per_group = df.groupby(group).size().tolist()
    max_perm = factorial(num_groups)/np.prod(factorial(np.array(num_per_group)))

    return max_perm

def check_is_paired(df, subject, group):
    is_pair = False
    if subject is not None:
        count_subject_groups = df.groupby(subject)[group].count()
        is_pair = (count_subject_groups > 1).any()
        
    return is_pair


def run_dabest(df, drop_cols=['sample'], subject='subject', group='group', test='mean_diff'):
    scores = pd.DataFrame()
    paired = False
    if subject is not None: 
        paired = check_is_paired(df, subject, group)
    
    groups = df[group].unique()
    
    if len(groups) == 2:
        df = df.set_index([subject,group])
        df = df.drop(drop_cols, axis=1)
        for col in df.columns:
            result = calculate_dabest(df.reset_index(), idx=(groups[0],groups[1]), x=group, y=col, id_col=subject, paired=paired)
            if scores.empty:
                scores = result
            else:
                scores = scores.append(result)
        scores = scores.set_index('identifier')
        
    return scores

def run_anova(df, alpha=0.05, drop_cols=["sample",'subject'], subject='subject', group='group', permutations=50):
    if subject is not None and check_is_paired(df, subject, group):
        groups = df[group].unique()
        drop_cols = [d for d in drop_cols if d != subject]
        if len(df[subject].unique()) == 1:
            res = run_ttest(df, groups[0], groups[1], alpha = alpha, drop_cols=drop_cols, subject=subject, group=group, paired=True, correction='indep', permutations=permutations)
        else:
            
            res = run_repeated_measurements_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, group=group, permutations=0)
    else:
        df = df.set_index([group])
        df = df.drop(drop_cols, axis=1)
        aov_results = []
        pairwise_results = []
        for col in df.columns:
            rows = df[col]
            aov_results.append((col,) + calculate_anova(rows, group=group))
            pairwise_results.append(calculate_THSD(rows, group=group))
            
        max_perm = get_max_permutations(df, group=group)
        res = format_anova_table(df, aov_results, pairwise_results,  group, permutations, alpha, max_perm)
    
    return res

def run_repeated_measurements_anova(df, alpha=0.05, drop_cols=['sample'], subject='subject', group='group', permutations=50):
    df = df.set_index([subject,group])
    df = df.drop(drop_cols, axis=1).dropna(axis=1)
    aov_results = []
    pairwise_results = []
    for col in df.columns:
        aov = calculate_repeated_measures_anova(df.reset_index(), column=col, subject=subject, group=group, alpha=alpha)
        aov_results.append(aov)
        pairwise_results.append(calculate_pairwise_ttest(df[col].reset_index(), column=col, subject=subject, group=group)) 
        
    max_perm = get_max_permutations(df, group=group)
    res = format_anova_table(df, aov_results, pairwise_results, group, permutations, alpha, max_perm)
    
    return res

def format_anova_table(df, aov_results, pairwise_results, group, permutations, alpha, max_permutations):
    columns = ['identifier', 'F-statistics', 'pvalue']
    scores = pd.DataFrame(aov_results, columns = columns)
    scores = scores.set_index('identifier')
       
    #FDR correction
    if permutations > 0 and max_permutations>=10:
        if max_permutations < permutations:
            permutations = max_permutations
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group=group, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
    
    res = pd.concat(pairwise_results)
    if not res.empty:
        res = res.join(scores[['F-statistics', 'pvalue', 'padj']].astype('float'))
        res['correction'] = scores['correction']
    else:
        res = scores
        res["log2FC"] = np.nan

    res = res.reset_index()
    res['rejected'] = res['padj'] < alpha
    res['-log10 pvalue'] = res['padj'].apply(lambda x: - np.log10(x))
    
    return res

def run_ttest(df, condition1, condition2, alpha = 0.05, drop_cols=["sample"], subject='subject', group='group', paired=False, correction='indep', permutations=50):
    columns = ['T-statistics', 'pvalue', 'mean_group1', 'mean_group2', 'log2FC']
    df = df.set_index([group, subject])
    df = df.drop(drop_cols, axis = 1)
    if paired:
        scores = df.T.apply(func = calculate_paired_ttest, axis=1, result_type='expand', args =(condition1, condition2))
    else:
        scores = df.T.apply(func = calculate_ttest, axis=1, result_type='expand', args =(condition1, condition2))
    scores.columns = columns
    scores = scores.dropna(how="all")

    max_perm = get_max_permutations(df, group=group)
    #FDR correction
    if permutations > 0:
        if max_perm < permutations:
            permutations = max_perm
        observed_pvalues = scores.pvalue
        count = apply_pvalue_permutation_fdrcorrection(df, observed_pvalues, group=group, alpha=alpha, permutations=permutations)
        scores= scores.join(count)
        scores['correction'] = 'permutation FDR ({} perm)'.format(permutations)
    else:
        rejected, padj = apply_pvalue_fdrcorrection(scores["pvalue"].tolist(), alpha=alpha, method = 'indep')
        scores['correction'] = 'FDR correction BH'
        scores['padj'] = padj
        scores['rejected'] = rejected
    #scores['rejected'] = scores['padj'] <= alpha
    scores['group1'] = condition1
    scores['group2'] = condition2
    scores['FC'] = [np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)) for x in scores['log2FC'].values]
    scores['-log10 pvalue'] = [- np.log10(x) for x in scores['padj'].values]
    scores = scores.reset_index() 

    return scores

def run_samr(df, subject='subject', group='group', drop_cols=['subject', 'sample'], alpha=0.05, s0=1, permutations=250):
    if permutations > 0:
        R_function = R('''result <- function(data, res_type, s0, nperms) {
                                    samr(data=data, resp.type=res_type, s0=s0, nperms=nperms, random.seed = 12345, s0.perc=NULL)
                                    }''')

        paired = check_is_paired(df, subject, group)
        groups = df[group].unique()
        samples = len(df[group])
        df = df.set_index(group).drop(drop_cols, axis=1).T

        labels = []
        conditions = set(df.columns)
        d = {v:k+1 for k, v in enumerate(conditions)}
        labels = [d.get(item,item)  for item in df.columns]

        if subject is not None:
            if len(groups) == 1:
                method = 'One class'
            elif len(groups) == 2:
                if paired:
                    method = 'Two class paired'
                    labels = []
                    counts = {}
                    conditions = df.columns.unique()
                    for col in df.columns:
                        cur_count = counts.get(col, 0)
                        labels.append([(cur_count+1)*-1 if col == conditions[0] else (cur_count+1)][0])
                        counts[col] = cur_count + 1
                else:
                    method = 'Two class unpaired'
            else:
                method = 'Multiclass'        

        delta = alpha
        data = base.list(x=base.as_matrix(df.values), y=base.unlist(labels), geneid=base.unlist(df.index), logged2=True)
        samr_res = R_function(data=data, res_type=method, s0=s0, nperms=permutations)
        delta_table = samr.samr_compute_delta_table(samr_res)
        siggenes_table = samr.samr_compute_siggenes_table(samr_res, delta, data, delta_table, all_genes=True)
        nperms_run = samr_res[8][0]
        f_stats = samr_res[9]
        pvalues = samr.samr_pvalues_from_perms(samr_res[9], samr_res[21])

        if isinstance(siggenes_table[0], np.ndarray):
            up = pd.DataFrame(np.reshape(siggenes_table[0], (-1, siggenes_table[3][0]))).T
        else:
            up = pd.DataFrame()
        if isinstance(siggenes_table[1], np.ndarray):
            down = pd.DataFrame(np.reshape(siggenes_table[1], (-1, siggenes_table[4][0]))).T
        else:
            down = pd.DataFrame()

        total = pd.concat([up, down])
        total = total.sort_values(total.columns[-1]).reset_index(drop=True)
        qvalues = total.iloc[:, -1].astype(float)/100
        qvalues = pd.DataFrame(qvalues)
        qvalues.insert(0, 'identifier', total[2])
        qvalues.columns = ['identifier', 'padj']
        
        if method == 'One class':
        #     res.columns = ['identifier', 't-statistics', 'pvalue', 'padj']
            df2 = pd.DataFrame()

        elif method == 'Multiclass':
            pairwise_results = []
            for col in df.T.columns:
                rows = df.T[col]
                pairwise_results.append(calculate_THSD(rows, group=group))
            pairwise = pd.concat(pairwise_results)
            
            res = pd.DataFrame([df.index, f_stats, pvalues]).T
            res.columns = ['identifier', 'F-statistics', 'pvalue']
            res = pairwise.join(res.set_index('identifier')).reset_index()

            contrasts = ['diff_mean_group{}'.format(str(i+1)) for i in np.arange(len(set(labels)))]
            df2 = total.iloc[:, 6:-1].reset_index(drop=True)
            df2.insert(0, 'id', total[2])
            df2.columns = ['identifier'] + contrasts

        else:
            log2FC = samr_res[10]
            res = pd.DataFrame([log2FC, f_stats, pvalues]).T
            res.index = df.index
            group1 = groups[0]
            group2 = groups[1]
            mean1 = df[group1].mean(axis=1)
            mean2 = df[group2].mean(axis=1)
            
            res = res.join(pd.DataFrame([mean1, mean2]).T, rsuffix='1').reset_index()
            res.columns = ['identifier', 'log2FC', 't-statistics', 'pvalue', 'mean(group1)', 'mean(group2)']
            res['group1'] = group1
            res['group2'] = group2
            res['FC'] = [np.power(2,np.abs(x)) * -1 if x < 0 else np.power(2,np.abs(x)) for x in res['log2FC'].values]
            res = res[['identifier', 'group1', 'group2', 'mean(group1)', 'mean(group2)', 'log2FC', 'FC', 't-statistics', 'pvalue']]
            df2 = pd.DataFrame()

        res = res.set_index('identifier').join(qvalues.set_index('identifier'))
        res['correction'] = 'permutation FDR ({} perm)'.format(nperms_run)
        res['-log10 pvalue'] = [- np.log10(x) for x in res['pvalue'].values]
        res['rejected'] = res['padj'] < 0.05
        res['Method'] = 'SAMR {}'.format(method)
        res = res.reset_index()
    else:
        res = run_anova(df, alpha=alpha, drop_cols=drop_cols, subject=subject, group=group, permutations=permutations)

    return res#, df2


def run_fisher(group1, group2, alternative='two-sided'):
    '''         annotated   not-annotated
        group1      a               b
        group2      c               d
        ------------------------------------

        group1 = [a, b]
        group2 = [c, d]

        odds, pvalue = stats.fisher_exact([[a, b], [c, d]])
    '''

    odds, pvalue = stats.fisher_exact([group1, group2], alternative)

    return (odds, pvalue)

def run_regulation_enrichment(regulation_data, annotation, identifier='identifier', groups=['group1', 'group2'], annotation_col='annotation', reject_col='rejected', group_col='group', method='fisher'):
    foreground_list = regulation_data[regulation_data[reject_col]][identifier].unique().tolist()
    background_list = regulation_data[~regulation_data[reject_col]][identifier].unique().tolist()
    grouping = []
    for i in annotation[identifier]:
        if i in foreground_list:
            grouping.append('foreground')
        elif i in background_list:
            grouping.append('background')
        else:
            grouping.append(np.nan)
    annotation[group_col] = grouping
    annotation = annotation.dropna(subset=[group_col])

    result = run_enrichment(annotation, foreground='foreground', background='background', foreground_pop=len(foreground_list), background_pop=len(background_list), annotation_col=annotation_col, group_col=group_col, identifier_col=identifier, method=method)
    
    
    return result

def run_enrichment(data, foreground, background, foreground_pop, background_pop, annotation_col='annotation', group_col='group', identifier_col='identifier', method='fisher'):
    result = pd.DataFrame()
    df = data.copy()
    terms = []
    ids = []
    pvalues = []
    fnum = []
    bnum = []
    countsdf = df.groupby([annotation_col,group_col]).agg(['count'])[(identifier_col,'count')].reset_index()
    countsdf.columns = [annotation_col, group_col, 'count']
    for annotation in countsdf[countsdf[group_col] == foreground][annotation_col].unique().tolist():
        counts = countsdf[countsdf[annotation_col] == annotation]
        num_foreground = counts.loc[counts[group_col] == foreground,'count'].values
        num_background = counts.loc[counts[group_col] == background,'count'].values
        
        if len(num_foreground) == 1:
            num_foreground = num_foreground[0]
        if len(num_background) == 1:
            num_background = num_background[0]
        else:
            num_background=0
        if method == 'fisher':
            odds, pvalue = run_fisher([num_foreground, foreground_pop-num_foreground],[num_background, background_pop-num_background])
        fnum.append(num_foreground)
        bnum.append(num_background)
        terms.append(annotation)
        pvalues.append(pvalue)
        ids.append(",".join(df.loc[(df[annotation_col]==annotation) & (df[group_col] == foreground), identifier_col].tolist()))
    if len(pvalues) > 1:
        rejected, padj = apply_pvalue_fdrcorrection(pvalues, alpha=0.05, method='indep')
        result = pd.DataFrame({'terms':terms, 'identifiers':ids, 'foreground':fnum, 'background':bnum, 'pvalue':pvalues, 'padj':padj, 'rejected':rejected})
        #result = result[result.rejected]
    return result

def calculate_fold_change(df, condition1, condition2):
    group1 = df[condition1]
    group2 = df[condition2]

    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values

    if np.isnan(group1).all() or np.isnan(group2).all():
        fold_change = np.nan
    else:
        fold_change = np.nanmedian(group1) - np.nanmedian(group2)

    return fold_change

def cohen_d(df, condition1, condition2, ddof = 0):
    group1 = df[condition1]
    group2 = df[condition2]

    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values

    ng1 = group1.size
    ng2 = group2.size
    dof = ng1 + ng2 - 2
    if np.isnan(group1).all() or np.isnan(group2).all():
        d = np.nan
    else:
        meang1 = np.nanmean(group1)
        meang2 = np.nanmean(group2)
        sdg1 = np.nanstd(group1, ddof = ddof)
        sdg2 = np.nanstd(group2, ddof = ddof)
        d = (meang1 - meang2) / np.sqrt(((ng1-1)* sdg1 ** 2 + (ng2-1)* sdg2 ** 2) / dof)

    return d

def hedges_g(df, condition1, condition2, ddof = 0):
    group1 = df[condition1]
    group2 = df[condition2]

    if isinstance(group1, np.float64):
        group1 = np.array(group1)
    else:
        group1 = group1.values
    if isinstance(group2, np.float64):
        group2 = np.array(group2)
    else:
        group2 = group2.values


    ng1 = group1.size
    ng2 = group2.size
    dof = ng1 + ng2 - 2
    if np.isnan(group1).all() or np.isnan(group2).all():
        g = np.nan
    else:
        meang1 = np.nanmean(group1)
        meang2 = np.nanmean(group2)
        sdpooled = np.nanstd(np.concatenate([group1, group2]), ddof = ddof)

        #Correct bias small sample size
        if ng1+ng2 < 50:
            g = ((meang1 - meang2) / sdpooled) * ((ng1+ng2-3) / (ng1+ng2-2.25)) * np.sqrt((ng1+ng2-2) / (ng1+ng2))
        else:
            g = ((meang1 - meang2) / sdpooled)

    return g

def run_mapper(data, lenses=["l2norm"], n_cubes = 15, overlap=0.5, n_clusters=3, linkage="complete", affinity="correlation"):
    X = data._get_numeric_data()
    labels ={i:data.index[i] for i in range(len(data.index))}

    model = ensemble.IsolationForest(random_state=1729)
    model.fit(X)
    lens1 = model.decision_function(X).reshape((X.shape[0], 1))

    # Create another 1-D lens with L2-norm
    mapper = km.KeplerMapper(verbose=0)
    lens2 = mapper.fit_transform(X, projection=lenses[0])

    # Combine both lenses to get a 2-D [Isolation Forest, L^2-Norm] lens
    lens = np.c_[lens1, lens2]

    # Define the simplicial complex
    simplicial_complex = mapper.map(lens,
                      X,
                      nr_cubes=n_cubes,
                      overlap_perc=overlap,
                      clusterer=cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity=affinity))



    return simplicial_complex, {"labels":labels}

def run_WGCNA(data, drop_cols_exp, drop_cols_cli, RsquaredCut=0.8, networkType='unsigned', minModuleSize=30, deepSplit=2, pamRespectsDendro=False, merge_modules=True, MEDissThres=0.25, verbose=0):
    """ 
    Runs an automated weighted gene co-expression network analysis (WGCNA), using input proteomics/transcriptomics/genomics and clinical variables data.

    Args:
        data: 
        drop_cols_exp: list of strings for columns in experimental dataset to be removed.
        drop_cols_cli: list of strings for columns in clinical dataset to be removed.
        RsquaredCut: desired minimum scale free topology fitting index R^2.
        networkType: network type ('unsigned', 'signed', 'signed hybrid', 'distance').
        minModuleSize: minimum module size.
        deepSplit: provides a rough control over sensitivity to cluster splitting, the higher the value (with 'hybrid' method) or if True (with 'tree' method), the more and smaller modules.
        pamRespectsDendro: only used for method 'hybrid'. Objects and small modules will only be assigned to modules that belong to the same branch in the dendrogram structure.
        merge_modules: if True, very similar modules are merged.
        MEDissThres: maximum dissimilarity (i.e., 1-correlation) that qualifies modules for merging.
        verbose: integer level of verbosity. Zero means silent, higher values make the output progressively more and more verbose.
    
    Returns:
        Tuple with multiple pandas dataframes.
    """
    result = {}
    dfs = wgcna.get_data(data, drop_cols_exp=drop_cols_exp, drop_cols_cli=drop_cols_cli)
    if 'clinical' in dfs:
        data_cli = dfs['clinical']   #Extract clinical data
        for dtype in dfs:
            if dtype in ['proteomics', 'rnaseq']:
                data_exp = dfs[dtype]   #Extract experimental data
                dtype = 'wgcna-'+dtype
                result[dtype] = {}

                softPower = wgcna.pick_softThreshold(data_exp, RsquaredCut=RsquaredCut, networkType=networkType, verbose=verbose)
                
                dissTOM, moduleColors = wgcna.build_network(data_exp, softPower=softPower, networkType=networkType, minModuleSize=minModuleSize, deepSplit=deepSplit,
                                                    pamRespectsDendro=pamRespectsDendro, merge_modules=merge_modules, MEDissThres=MEDissThres, verbose=verbose)

                Features_per_Module = wgcna.get_FeaturesPerModule(data_exp, moduleColors, mode='dataframe')

                MEs = wgcna.calculate_module_eigengenes(data_exp, moduleColors, softPower=softPower, dissimilarity=False)

                moduleTraitCor, textMatrix = wgcna.calculate_ModuleTrait_correlation(data_exp, data_cli, MEs)

                MM, MMPvalue = wgcna.calculate_ModuleMembership(data_exp, MEs)

                FS, FSPvalue = wgcna.calculate_FeatureTraitSignificance(data_exp, data_cli)

                METDiss, METcor = wgcna.get_EigengenesTrait_correlation(MEs, data_cli)

                result[dtype]['dissTOM'] = dissTOM
                result[dtype]['module_colors'] = moduleColors
                result[dtype]['features_per_module'] = Features_per_Module
                result[dtype]['MEs'] = MEs
                result[dtype]['module_trait_cor'] = moduleTraitCor
                result[dtype]['text_matrix'] = textMatrix
                result[dtype]['module_membership'] = MM
                result[dtype]['module_membership_pval'] = MMPvalue
                result[dtype]['feature_significance'] = FS
                result[dtype]['feature_significance_pval'] = FSPvalue
                result[dtype]['ME_trait_diss'] = METDiss
                result[dtype]['ME_trait_cor'] = METcor
    
    return result

def most_central_edge(G):
    centrality = nx.eigenvector_centrality_numpy(G, weight='width')

    return max(centrality, key=centrality.get)

def get_louvain_partitions(G, weight):
    partition = community.best_partition(G)

    return partition

def get_network_communities(graph, args):
    if 'communities_algorithm' not in args:
        args['communities_algorithm'] = 'louvain'

    if args['communities_algorithm'] == 'louvain':
        communities = get_louvain_partitions(graph, args['values'])
    elif args['communities_algorithm'] == 'greedy_modularity':
        gcommunities = nx.algorithms.community.greedy_modularity_communities(graph, weight=args['values'])
        communities = utils.generator_to_dict(gcommunities)
    elif args['communities_algorithm'] == 'asyn_label_propagation':
        gcommunities = nx.algorithms.community.label_propagation.asyn_lpa_communities(graph, args['values'])
        communities = utils.generator_to_dict(gcommunities)
    elif args['communities_algorithm'] == 'girvan_newman':
        gcommunities = nx.algorithms.community.girvan_newman(graph, most_valuable_edge=most_central_edge)
        communities = utils.generator_to_dict(gcommunities)
    elif args['communities_algorithm'] == 'affinity_propagation':
        adjacency = nx.to_pandas_adjacency(graph, weight='width')
        nodes = list(adjacency.columns)
        communities = AffinityPropagation().fit(adjacency.values).labels_
        communities = {nodes[i]:communities[i] for i in range(len(communities))}


    return communities

def get_publications_abstracts(data, publication_col="publication", join_by=['publication','Proteins','Diseases'], index="PMID"):
    abstracts = pd.DataFrame()
    if not data.empty:
        abstracts = utils.getMedlineAbstracts(list(data.reset_index()[publication_col].unique()))
        if not abstracts.empty:
            abstracts = abstracts.set_index(index)
            abstracts = abstracts.join(data.reset_index()[join_by].set_index(publication_col)).reset_index()

    return abstracts

def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov

def omega_squared(aov):
    mse = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    return aov

def run_two_way_anova(df, drop_cols=['sample'], subject='subject', group=['group', 'secondary_group']):
    """ 
    Run a 2-way ANOVA when data['secondary_group'] is not empty

    Args:
        df: processed pandas dataframe with samples as rows, and proteins and groups as columns.
        drop_cols: list of column names to drop from dataframe
        subject:
        group: list of column names corresponding to independent variable groups

    """ 

    data = df.copy()
    factorA, factorB = group
    data = data.set_index([subject]+group)
    data = data.drop(drop_cols, axis=1)
    data.columns = data.columns.str.replace(r"-", "_")

    aov_result = []
    residuals = {}
    for col in data.columns:
        model = ols('{} ~ C({})*C({})'.format(col, factorA, factorB), data[col].reset_index().sort_values(group, ascending=[True, False])).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        eta_squared(aov_table)
        omega_squared(aov_table)
        for i in aov_table.index:
            if i != 'Residual':
                t, p, eta, omega = aov_table.loc[i, ['F', 'PR(>F)', 'eta_sq', 'omega_sq']]
                protein = col.replace('_', '-')
                aov_result.append((protein, i, t, p, eta, omega))
        residuals[col] = model.resid

    anova_df = pd.DataFrame(aov_result, columns = ['identifier','source', 'F-statistics', 'pvalue', 'eta_sq', 'omega_sq'])
    anova_df = anova_df.set_index('identifier')
    anova_df = anova_df.dropna(how="all")
    
    return anova_df, residuals