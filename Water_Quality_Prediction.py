import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from pandas.errors import SettingWithCopyWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report,confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, cross_validate, validation_curve,cross_val_score
import warnings

from sklearn.naive_bayes import GaussianNB

warnings.simplefilter(action="ignore")
warnings.filterwarnings("ignore", message="Found whitespace in feature_names, replace with underlines")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import confusion_matrix,auc
from sklearn.naive_bayes import GaussianNB

pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
df = pd.read_csv("water_potability.csv")
df.head(20)
df.info()

#############################################
# 1.Exploratory Data Analysis
#############################################

#############################################
# General Info
#############################################

def check_df(dataframe, head=5):
    print("########## Shape ###########")
    print(dataframe.shape)
    print("########## Type #############")
    print(dataframe.dtypes)
    print("########## Head #############")
    print(dataframe.head(head))
    print("########## Tail #############")
    print(dataframe.tail(head))
    print("############ NA ##############")
    print(dataframe.isnull().sum())
    print("########## Quartiles #########")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T.round(4))
    print("########## Unique #########")
    print(dataframe.nunique())

check_df(df)

#############################################
# Numerik ve kategorik değişkenler
#############################################
def grab_col_names(dataframe,cat_th=2,car_th=20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]

    num_cols=[col for col in df.columns if df[col].dtypes in ["float","int"]]
    num_cols=[col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)} ")
    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car = grab_col_names(df)


#############################################
# Numerik ve kategorik değişkenlerin analizi
#############################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()/len(dataframe)}))
    print("################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):

    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(8, 6))
        dataframe[numerical_col].hist(bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel(numerical_col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {numerical_col}')
        plt.grid(False)
        plt.tight_layout()
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

num_summary(df,num_cols,plot=True)

#############################################
# Analysis of Target Variable
#############################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Potability", col)

#############################################
# Analysis of Correlation
#############################################
def heatmap_correlation(dataframe, numerical_cols):
    corr = dataframe[numerical_cols].corr()
    colors=["#225FC6","#386DC6","#537CC2","#6988BD","#89A1CB","#AABDDD","#C5D0E4","#DCE0E8"]
    f, ax = plt.subplots(figsize=[10, 5], facecolor='#9FC1F4')
    sns.heatmap(dataframe[numerical_cols].corr(), annot=True, fmt=".3f", ax=ax, cmap=colors, linewidths=0.5, linecolor="lightskyblue", vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix", fontsize=14, color="steelblue", weight="bold", pad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', color="#FAF7F2")
    ax.set_yticklabels(ax.get_yticklabels(), color="#FAF7F2")
    plt.tight_layout()
    plt.show()

# Kullanım
heatmap_correlation(df, num_cols)

################################################
# Visualizing distributions of variables
################################################

potable = df.query("Potability == 1")
non_potable = df.query("Potability == 0")
def plot_distributions(dataframe, not_potable_df, potable_df):
    fig = plt.figure(figsize=(20, 10), facecolor="#FAF7F2")
    for i, column in enumerate(dataframe.columns[:9]):
        plt.subplot(3, 3, i + 1)
        plt.title(f"Distribution of {column} values", color="steelblue", weight='bold')
        sns.kdeplot(x=not_potable_df[column], color="navy", label="Not Potable(0)")
        sns.kdeplot(x=potable_df[column], color="cornflowerblue", label="Potable(1)")
        plt.xlabel(column, color="grey")
        plt.ylabel("Density", color="grey")
        plt.legend(prop=dict(size=10))
    plt.tight_layout()
    plt.show()


plot_distributions(df, non_potable, potable)

#Distribution of target variable classes

def plot_distributions_class(dataframe, not_potable_df, potable_df):
    fig = plt.figure(figsize=(20, 10), facecolor="#9FC1F4")
    for i, column in enumerate(dataframe.columns[:9]):
        plt.subplot(3, 3, i + 1)
        plt.title(f"Distribution of {column} values", color="steelblue", weight='bold')
        sns.distplot(not_potable_df[column], label="Not Potable(0)", color="navy", hist_kws=dict(edgecolor="k", linewidth=1), bins=25)
        sns.distplot(potable_df[column], label='Potable(1)', color="#9FC1F4", hist_kws=dict(edgecolor='k', linewidth=1), bins=25)
        plt.xlabel(column, color="grey")
        plt.ylabel("Density", color="grey")
        plt.legend(prop=dict(size=10))
    plt.tight_layout()
    plt.show()

plot_distributions_class(df, non_potable, potable)


# Pie chart distribution of the target variable
def plot_potability_distribution(potability_counts):
    plt.figure(figsize=(6, 6), facecolor='#FAF7F2')
    plt.pie(potability_counts, labels=potability_counts.index, autopct='%1.1f%%', startangle=140, colors=["navy", "cornflowerblue"], textprops={'color': 'white', 'fontweight': 'bold'})
    plt.title('Potability Distribution', color="steelblue", weight='bold')
    plt.axis('equal')
    plt.text(-0.32, -0.7, 'Not Potable', color='white', weight='bold')
    plt.text(0.10, 0.45, 'Potable', color='white', weight='bold')
    plt.show()

plot_potability_distribution(df['Potability'].value_counts())


#############################################
# 2.Data Preprocessing & Feature Engineering
#############################################

#############################################
# Outliers
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

#Boxplot ile ayrkırı değerleri görselleştirme
plt.figure(figsize=(5, 8), facecolor='#FAF7F2')

for i, col in enumerate(num_cols):
    plt.subplot(4, 2, i+1)
    sns.boxplot(x=df[col], palette="Set3",whis=1.5)
    plt.title(col, fontsize=12, color="steelblue", weight="bold", pad=10)
    plt.xlabel("Value", fontsize=10, color="#333333")
    plt.ylabel("Count", fontsize=10, color="#333333")

plt.tight_layout()
plt.show()

def plot_boxplots(dataframe, numerical_cols):
    plt.figure(figsize=(5, 8), facecolor='#FAF7F2')

    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 2, i+1)
        sns.boxplot(x=dataframe[col], palette="Set3", whis=1.5)
        plt.title(col, fontsize=12, color="steelblue", weight="bold", pad=10)
        plt.xlabel("Value", fontsize=10, color="#333333")
        plt.ylabel("Count", fontsize=10, color="#333333")
        plt.tight_layout()
    plt.show()


plot_boxplots(df, num_cols)

#############################################
# Missing Values
#############################################

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns
missing_values_table(df)

# Eksik değerler için görselleştirme
def plot_missing_data(dataframe):
    plt.figure(figsize=(12, 6), facecolor='#9FC1F4')
    sns.barplot(x=dataframe.columns, y=dataframe.isnull().sum(), palette="Set3")
    plt.bar(dataframe.columns, dataframe.shape[0], color='lightgray', alpha=0.5, label="Toplam Gözlem Sayısı")
    plt.title('Eksik Veri ve Toplam Gözlem Sayısı', fontsize=14, color='steelblue', weight='bold')
    plt.xlabel('Değişkenler', fontsize=12, color='w', labelpad=10)
    plt.ylabel('Veri Sayısı', fontsize=12, color='w', labelpad=10)
    plt.xticks(rotation=45, ha='right', color='w')
    plt.yticks(color='w')
    plt.legend()
    for i, val in enumerate(dataframe.isnull().sum()):
        plt.text(i, val + 5, str(val), color='#333333', ha='center')
    plt.tight_layout()
    plt.show()
plot_missing_data(df)

#Eksik verilerin medyan ile doldurulması
df['ph'].replace(0, np.nan, inplace=True)
df["Sulfate"].fillna(df["Sulfate"].median(), inplace=True)
df["ph"].fillna(df["ph"].median(), inplace=True)
df["Trihalomethanes"].fillna(df["Trihalomethanes"].median(), inplace=True)

#############################################
# Feature Extraction 
#############################################

# ph, sülfat, iletkenlik değerlerini içmesuyu kalite standartlarının limitlerine göre gruplandırma
df['NEW_PH_CAT'] = pd.cut(df['ph'], bins=[0, 6.5, 9.5, 14], labels=['Düşük_pH', 'Uygun_pH', 'Yüksek_pH'])
df['NEW_SULFATE_CAT'] = pd.cut(df['Sulfate'], bins=[0, 250, float("inf")], labels=["Uygun", "LimitAşımı"])
df['NEW_CONDUCTIVITY_CAT'] = pd.cut(df['Conductivity'], bins=[0, 400, float("inf")], labels=["Uygun", "LimitAşımı"])
df['NEW_TURBIDITY_CAT'] = pd.cut(df['Turbidity'], bins=[-float("inf"), 5, float("inf")], labels=["Uygun", "LimitAşımı"])

#df['NEW_HARDNESS_CAT'] = pd.cut(df['Hardness'], bins=[0, 500, float("inf")], labels=["Uygun", "LimitAşımı"])
#df['NEW_TDS_CAT'] = pd.cut(df['Solids'], bins=[0, 500, 1000, float("inf")], labels=["AltLimitAşımı","Uygun","LimitAşımı"])
#df['NEW_CHLORAMINES_CAT'] = pd.cut(df["Chloramines"], bins=[0, 4 , float("inf")], labels=["Uygun","LimitAşımı"])
#df['NEW_ORGANIC_CARBON_CAT'] = pd.cut(df['Organic_carbon'], bins=[0, 4, float("inf")], labels=["Uygun", "LimitAşımı"])
#df['NEW_TRIHALOMETHANES_CAT'] = pd.cut(df['Trihalomethanes'], bins=[-float("inf"), 100, float("inf")], labels=["Uygun", "LimitAşımı"])

#Yüksek pH değerlerinde yüksek THM oluşumu gözleniyormuş. Bu verileri birleştirerek yeni değişken oluşturuldu.
df['NEW_THM_PH_SUM']= df['ph'] + df['Trihalomethanes']

#TDS (Toplam Çözünmüş Katılar) / EC (Elektriksel İletkenlik) oranı
df['NEW_TDS_CON_RATIO']= df['Solids']/df['Conductivity']

#Bulanıklık ve toplam çözünmüş maddelerin çarpımı
df['NEW_TURBIDITY_SOLIDS'] = df['Turbidity'] * (df['Solids'] + df['Conductivity'])

# Klor kullanımı sonucunda oluşan THM'nin ve kloramin miktarları toplamı
df['NEW_CHLOR_SUM'] = df['Trihalomethanes']+df["Chloramines"]

#############################################
# One-Hot Encoding
#############################################
cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in ["Potability"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype=bool)
    dataframe = dataframe.astype(int)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()



#############################################
# Feature Scaling 
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


#######################
#SMOTE
######################

y = df["Potability"]
X = df.drop(["Potability"], axis=1)

before_counts = Counter(y)
total_before = sum(before_counts.values())
smote = SMOTE()
X, y = smote.fit_resample(X, y)
after_counts = Counter(y)
total_after = sum(after_counts.values())

def plot_smote_class_distribution(before_counts, after_counts, total_before, total_after):
    plt.figure(figsize=(6, 6), facecolor='#9FC1F4')
    plt.bar([str(label) for label in before_counts.keys()], before_counts.values(), color='b', alpha=0.6, label=f'Önce ({total_before} Toplam)')
    plt.bar([str(label) for label in after_counts.keys()], after_counts.values(), color="cornflowerblue", alpha=0.6, label=f'Sonra ({total_after} Toplam)')

    for i, count in enumerate(before_counts.values()):
        plt.text(i, count, str(count), ha='center', va='bottom', color="navy")

    for i, count in enumerate(after_counts.values()):
        plt.text(i, count, str(count), ha='center', va='bottom', color="navy")

    plt.title('Sınıf Dağılımı Öncesi ve Sonrası SMOTE', color="navy", weight='bold')
    plt.xlabel('Sınıf Etiketleri', color="teal", weight='bold')
    plt.ylabel('Örnek Sayısı', color="teal", weight='bold')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_smote_class_distribution(before_counts, after_counts, total_before, total_after)


#############################################
# Base Models
#############################################

def base_models(X, y, scoring="roc_auc"):
    print("Base Models...")
    classifiers = [('LR', LogisticRegression()),
                   ('CART', DecisionTreeClassifier()),
                   ('RF', RandomForestClassifier()),
                   ('SVC', SVC()),
                   ('GBM', GradientBoostingClassifier()),
                   ("XGBoost", XGBClassifier(objective='reg:squarederror')),
                   ("LightGBM", LGBMClassifier(verbose=0)),
                   ("CatBoost", CatBoostClassifier(verbose=False)),
                   ("GaussianNB", GaussianNB())]

    results = []
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring)
        mean_score = round(cv_results['test_score'].mean(), 2)
        results.append((name, mean_score))

    df = pd.DataFrame(results, columns=['Model', scoring])
    return df

# ROC AUC değerlerini içeren tablo
roc_auc_table = base_models(X, y)
print(roc_auc_table)

# ROC AUC değerlerinin modellere göre kıyaslama grafiği
def plot_performance_metric(model_performance_table, metric='roc_auc'):
    plt.figure(figsize=(10, 6), facecolor='#9FC1F4')
    bars = plt.bar(model_performance_table['Model'], model_performance_table[metric], color='#225FC6', edgecolor="midnightblue", linewidth=1)
    plt.xlabel('Model', color='grey')
    plt.ylabel(metric.upper(), color='steelblue', fontsize=12, labelpad=10)
    plt.title(f'Model Performansı - {metric.upper()}', color="steelblue", weight="bold", pad=10)
    plt.xticks(rotation=45, color="teal")
    plt.yticks(color="cornflowerblue")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), va='bottom', ha='center', color='midnightblue', fontsize=12)

    plt.tight_layout()
    plt.show()

plot_performance_metric(roc_auc_table, metric="roc_auc")


# Doğruluk değerlerinin modellere göre kıyaslama grafiği
accuracy_table = base_models(X, y, scoring="accuracy")
print(accuracy_table)
plot_performance_metric(accuracy_table, metric='accuracy')

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
rf_model.get_params()
rf_params = {"max_depth": [20,None],
             "max_features": [3,5,'sqrt'],
             "min_samples_split": [1.5,2],
             "n_estimators": [100,200,600]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
cv_results_rf_final = cross_validate(rf_final,X,y, cv=5, scoring=["accuracy","f1","roc_auc","recall","precision"])
print(f"Accuracy: {round(cv_results_rf_final['test_accuracy'].mean(),2)}\n"
      f"Recall: {round(cv_results_rf_final['test_recall'].mean(),2)}\n"
      f"Precision: {round(cv_results_rf_final['test_precision'].mean(),2)}\n"
      f"F1 Score: {round(cv_results_rf_final['test_f1'].mean(),2)}\n"
      f"ROC AUC: {round(cv_results_rf_final['test_roc_auc'].mean(),2)}")


################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(objective='reg:squarederror')
xgboost_model.get_params()
xgboost_params = {"colsample_bytree": [0.9,1,"None"],
                   "learning_rate": [0.01, 0.07,0.09,"None"],
                   "n_estimators": [200,400,600]}

xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)
cv_results_xg_final = cross_validate(xgboost_final,X,y, cv=5, scoring=["accuracy","f1","roc_auc","recall","precision"])
print(f"Accuracy: {round(cv_results_xg_final['test_accuracy'].mean(),2)}\n"
      f"Recall: {round(cv_results_xg_final['test_recall'].mean(),2)}\n"
      f"Precision: {round(cv_results_xg_final['test_precision'].mean(),2)}\n"
      f"F1 Score: {round(cv_results_xg_final['test_f1'].mean(),2)}\n"
      f"ROC AUC: {round(cv_results_xg_final['test_roc_auc'].mean(),2)}")

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17, verbose=0)
lgbm_model.get_params()
lgbm_params = {"learning_rate": [0.05,0.1],
                "n_estimators": [200,600,700],
                "colsample_bytree": [0.5,0.9,1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=False).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
cv_results_lgbm_final = cross_validate(lgbm_final,X,y, cv=5, scoring=["accuracy","f1","roc_auc","recall","precision"])
print(f"Accuracy: {round(cv_results_lgbm_final['test_accuracy'].mean(),2)}\n"
      f"Recall: {round(cv_results_lgbm_final['test_recall'].mean(),2)}\n"
      f"Precision: {round(cv_results_lgbm_final['test_precision'].mean(),2)}\n"
      f"F1 Score: {round(cv_results_lgbm_final['test_f1'].mean(),2)}\n"
      f"ROC AUC: {round(cv_results_lgbm_final['test_roc_auc'].mean(),2)}")

################################################
# Catboost
################################################

catboost_model = CatBoostClassifier(verbose=False,random_state=17)
catboost_model.get_params()
catboost_params = {'learning_rate': [0.01, 0.02],
                   'depth': [7,12],
                   'iterations': [100,700]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
cv_results_cat_final = cross_validate(catboost_final,X,y, cv=5, scoring=["accuracy","f1","roc_auc","recall","precision"])
print(f"Accuracy: {round(cv_results_cat_final['test_accuracy'].mean(),2)}\n"
      f"Recall: {round(cv_results_cat_final['test_recall'].mean(),2)}\n"
      f"Precision: {round(cv_results_cat_final['test_precision'].mean(),2)}\n"
      f"F1 Score: {round(cv_results_cat_final['test_f1'].mean(),2)}\n"
      f"ROC AUC: {round(cv_results_cat_final['test_roc_auc'].mean(),2)}")

######################################################
#  Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [20,None],
             "max_features": [3,5,'sqrt'],
             "min_samples_split": [1.5,2],
             "n_estimators": [100,200,600]}

xgboost_params = {"colsample_bytree": [0.9,1,"None"],
                   "learning_rate": [0.01, 0.07,0.09,"None"],
                   "n_estimators": [200,400,600]}

lgbm_params = {"learning_rate": [0.05,0.1],
                "n_estimators": [200,600,700],
                "colsample_bytree": [0.5,0.9,1]}

catboost_params = {'learning_rate': [0.01, 0.02],
                   'iterations': [100,500]}

classifiers =  [('RF', RandomForestClassifier(),rf_params),
                ("LightGBM", LGBMClassifier(verbose=0),lgbm_params),
                ('XGBoost', XGBClassifier(objective='reg:squarederror'),xgboost_params),
                ("CatBoost", CatBoostClassifier(verbose=False),catboost_params)]

def hyperparameter_optimization(X,y,cv=5,scoring="roc_auc"):
    print("Hyperparameter Optimization...")
    best_models={}
    for name, classifier,params in classifiers:
        print(f"###{name}###")
        cv_results=cross_validate(classifier,X,y,cv=cv,scoring=scoring)
        print(f"{scoring}(Before):{round(cv_results['test_score'].mean(),3)}")

        gs_best=GridSearchCV(classifier,params,cv=cv,n_jobs=-1,verbose=False).fit(X,y)
        final_model=classifier.set_params(**gs_best.best_params_)

        cv_results=cross_validate(final_model,X,y,cv=cv,scoring=scoring)
        print(f"{scoring}(After):{round(cv_results['test_score'].mean(),3)}")
        print(f"{name}best_params:{gs_best.best_params_}",end="\n\n")
        best_models[name]=final_model
    return best_models

best_models=hyperparameter_optimization(X,y)

#################################
#Final Model
#################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modelin test seti üzerinde performansını değerlendirme
y_pred = rf_final.fit(X_train,y_train).predict(X_test)
y_pred_proba = rf_final.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": round(accuracy_score(y_test, y_pred), 2),
    "Recall": round(recall_score(y_test, y_pred), 2),
    "Precision": round(precision_score(y_test, y_pred), 2),
    "F1": round(f1_score(y_test, y_pred), 2),
    "ROC AUC": round(roc_auc_score(y_test, y_pred_proba), 2)
}
for metric, value in metrics.items():
    print(f"{metric}: {value}")


joblib.dump(rf_final,"rf_final.pkl")

########################
#Confusion Matrix
########################

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 5), facecolor='#FAF7F2')
    sns.set(font_scale=1.5)
    sns.heatmap(cm,
                xticklabels=['Potable', 'Non-Potable'],
                yticklabels=['Potable', 'Non-Potable'],
                annot=True,
                cmap='Blues',
                linewidths=0.5,
                fmt='d')
    plt.title("Hata Matrisi", color="#225FC6")
    plt.ylabel('Tahmin Sonuçları', color="#225FC6")
    plt.xlabel('Gerçek Sonuçlar', color="#225FC6")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_test, y_pred)

##################
# ROC-CURVE
##################
def plot_roc_curve(y_true, y_pred_proba):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    sns.set_theme(style='white')
    plt.figure(figsize=(8, 8), facecolor="#9FC1F4")
    plt.plot(false_positive_rate, true_positive_rate, color='#b01717', label='AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', color='#174ab0')
    plt.axis('tight')
    plt.ylabel('True Positive Rate', color="#225FC6")
    plt.xlabel('False Positive Rate', color="#225FC6")
    plt.show()

plot_roc_curve(y_test, y_pred_proba)


################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10,8),facecolor="#9FC1F4")
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",ascending=False)[0:num])
    plt.title('Feature Importance', color="#225FC6")
    plt.xticks(rotation=45, color="teal")
    plt.yticks(color="navy")
    plt.tight_layout()
    plt.show()

plot_importance(rf_final, X)


################################################
# Analyzing Model Complexity with Learning Curves
# ################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1],scoring="roc_auc")

rf_val_params[0][1]


################################################
# Stacking & Ensemble Learning
################################################
def voting_classifier(best_models,X,y):
    print("Voting Classifier...")
    voting_clf=VotingClassifier(estimators=[("RF",best_models["RF"]),
                                            ("CatBoost",best_models["CatBoost"]),
                                            ("XGBoost", best_models["XGBoost"])],voting="soft").fit(X,y)
    cv_results=cross_validate(voting_clf,X,y,cv=3,scoring=["accuracy","f1","roc_auc"])
    print(f"Accuracy:{cv_results['test_accuracy'].mean()}")
    print(f"F1Score:{cv_results['test_f1'].mean()}")
    print(f"ROC_AUC:{cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf=voting_classifier(best_models,X,y)

joblib.dump(voting_clf,"voting_clf.pkl")

