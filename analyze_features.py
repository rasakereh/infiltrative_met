from select import select
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from dfply import *
# from sklearn.tree import DecisionTreeClassifier
from scipy.stats.stats import pearsonr


plates_info = pd.read_csv('plates_info_dual.csv')
# print(plates_info)

# annots = pd.read_csv('../data/resection/annot.csv')
# annots.columns = annots.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('#', 'no')
# annots.study_case_no = annots.study_case_no.astype('Int32')
# annots = annots >> select(X.study_group, X.study_case_no)
# annots = annots >> mask(X.study_group.str.contains('LB'))

gradings = pd.read_csv('new_grades.csv')
gradings = gradings >> select(columns_between('case', 'total'))
gradings.domin_s = gradings.domin_s.astype('Int32')
gradings.high_s = gradings.high_s.astype('Int32')
gradings = gradings >> mask(~X.domin_s.isna())
# gradings.case = gradings.case.str.slice(5, 7).astype('Int32')

print('========================================')
print(plates_info['case'].unique())
print('========================================')
ft_grd_data = plates_info >> inner_join(gradings, by='case')# >> drop(X.mask_area, X.mask_perimeter, X.case)
print('========================================')
print(ft_grd_data['case'].unique())
print('========================================')
# ft_grd_data.set_index('domin_s', inplace=True)
# ft_grd_data = ft_grd_data.groupby(level='domin_s')
# ft_grd_data.boxplot(rot=45)
# plt.show()


reshaped_ft_grade = pd.melt(ft_grd_data, ['summary'] + list(gradings.columns))
reshaped_ft_grade['feature'] = reshaped_ft_grade['variable'] + '_' + reshaped_ft_grade['summary']
reshaped_ft_grade = reshaped_ft_grade >> drop(X.variable, X.summary)
reshaped_ft_grade = reshaped_ft_grade.pivot(index = list(gradings.columns), columns = 'feature', values='value').reset_index()
reshaped_ft_grade.to_csv('reshaped_dual.csv', index=False)

feature_vals = reshaped_ft_grade >> select(columns_from('total')) >> drop(X.total)
scores = reshaped_ft_grade >> select(X.domin_s, X.high_s, X.total)
na_highs = scores['high_s'].isna()
scores['high_s'][na_highs] = scores['domin_s'][na_highs]

# corr_X_Y = [[(type(feature_vals[feature]), type(scores[score])) for feature in feature_vals] for score in scores]
corr_X_Y_corr = [[pearsonr(feature_vals[feature], scores[score])[0] for score in scores] for feature in feature_vals]
corr_X_Y_pval = [[pearsonr(feature_vals[feature], scores[score])[1] for score in scores] for feature in feature_vals]
corr_X_Y_corr = pd.DataFrame(corr_X_Y_corr, index=feature_vals.columns, columns=scores.columns)
corr_X_Y_pval = pd.DataFrame(corr_X_Y_pval, index=feature_vals.columns, columns=scores.columns)

print(corr_X_Y_pval)
print('======================================================')
print(corr_X_Y_corr)
print('======================================================')

corr_X_Y_pval.to_csv('corr_X_Y_pval.csv')
corr_X_Y_corr.to_csv('corr_X_Y_corr.csv')


# from sklearn.decomposition import PCA
# pca = PCA(n_components=4)
# pca.fit(feature_vals.T)

# pca_res = pca.components_
# corr_X_Y = [pearsonr(col, reg_Y) for col in pca_res]
# print(*sorted(corr_X_Y, key=lambda x: x[1]), sep='\n')
