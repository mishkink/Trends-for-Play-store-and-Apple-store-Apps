import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, TransformerMixin

def visual(x, y, plot_type, title, xlabel, ylabel, rotation=False, rotation_value=60, figsize=(15, 8)):
    plt.figure(figsize=figsize)

    if plot_type == "bar":
        sns.barplot(x=x, y=y)
    elif plot_type == "count":
        sns.countplot(x)
    elif plot_type == "reg":
        sns.regplot(x=x, y=y)
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.yticks(fontsize=13)
    if rotation == True:
        plt.xticks(fontsize=13, rotation=rotation_value)
    plt.show()


data = pd.read_csv('Dataset/AppleStore.csv')
app_desc = pd.read_csv('Dataset/appleStore_description.csv')

print(data.head())

print(data.info())

print(app_desc.head())

# Top 10 apps on the basis of total rating

# total rating is an indicator of number of downloads so we will treat total rating count as a target variable

store_data_sorted = data.sort_values('rating_count_tot', ascending=False)
subset_store_data_sorted = store_data_sorted[:10]

print(visual(subset_store_data_sorted.track_name, subset_store_data_sorted.rating_count_tot, "bar", "Top 10 apps in apple store on the basis of ratings",
          "APP NAME", "RATING COUNT (TOTAL)", True, -60))

# Top 10 apps
data.currency.unique()
store_data_price = data.sort_values('price', ascending=False)
subset_store_data_price = store_data_price[:10]

print(visual(subset_store_data_price.price, subset_store_data_price.track_name, "bar", "Top 10 apps in apple store on the basis of price",
          "Price (in USD)", "APP NAME"))




print(visual(data["lang.num"], data.rating_count_tot, "reg",
          "Correlation of No. of Languages and Rating count", "NUMBER OF LANGAUGES",
          "RATING COUNT", False))


data['revenue'] = data.rating_count_tot * data.price
store_data_business = data.sort_values("revenue", ascending=False)
subset_store_data_business = store_data_business[:10]

visual(subset_store_data_business.track_name, subset_store_data_business['revenue'], "bar", "Highest revenue",
         "APP NAME", "REVENUE", True, -60)


# User Ratings on the App Store

visual(data.user_rating, None, "count","Ratings on App store",
         "RAITNGS", "NUMBER OF APPS RATED")

data["favourites_tot"] = data["rating_count_tot"] * data["user_rating"]
data["favourites_ver"] = data["rating_count_ver"] * data["user_rating_ver"]

favourite_app = data.sort_values("favourites_tot", ascending=False)
favourite_app_subset = favourite_app[:10]

visual(favourite_app_subset.track_name, favourite_app_subset.rating_count_tot, "bar", "User Favourites ",
         "APP NAME",  "RATING COUNT(TOTAL)", True, -60)


## !!!!!! Modeling !!!!!!
store_train, store_test = train_test_split(data, test_size=0.2)

data = store_train
data.info()

# Drops unncessary columns
class dropper(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        X = pd.DataFrame(X)
        return X.drop(["currency", "rating_count_tot", "rating_count_ver", "track_name",
                       "Unnamed: 0", "vpp_lic", "revenue",
                       "favourites_tot", "favourites_ver"], axis=1)


def ver_cleaner(data):
    try:
        if "V3" in data:
            return str(3)
        else:
            return int(data.split(".")[0])
    except:
        return int(0)


class version_trimmer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        X["ver"] = X["ver"].apply(ver_cleaner)
        return X

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

class dual_encoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self):
        pass

    def fit_transform(self, X, y=None):
        self.encoder_cont = LabelEncoder()
        cont_encoded = self.encoder_cont.fit_transform(X['cont_rating'])

        self.encoder_prime_genre = LabelEncoder()
        genre_encoded = self.encoder_prime_genre.fit_transform(X['prime_genre'])

        X["cont_encoded"] = cont_encoded
        X["genre_encoded"] = genre_encoded

        return X.drop(["cont_rating", "prime_genre"], axis=1)


def model_scoring(model_name, model, X, y):
    # Cross Validation
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)

    # Scores
    rmse = np.sqrt(-scores)
    mean = rmse.mean()
    std = rmse.std()
    print(model_name)
    print()
    print("RMSE: {}".format(rmse))
    print("MEAN: {}".format(mean))
    print("STD: {}".format(std))


# Data pipeline
category_attributes = ["cont_rating","prime_genre"]
numerical_attributes = data.drop(["cont_rating","prime_genre"], axis=1).columns

numline = Pipeline([("dataframe", DataFrameSelector(numerical_attributes)),
                    ("dropper", dropper()),
                    ("version-trimmer", version_trimmer()),
                   ("scaling",StandardScaler())])

encoder = dual_encoder()

catline = Pipeline([("dataframe", DataFrameSelector(category_attributes)),
                    ("cat-encoder", encoder)])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", numline),
                                               ("cat_pipeline", catline)])

store_data_prepared = full_pipeline.fit_transform(data)


#Encoders
cont_codes = encoder.encoder_cont.classes_
genre_codes = encoder.encoder_prime_genre.classes_

y = np.c_[data["rating_count_tot"]] #labels
X = store_data_prepared #Attributes
# Linear Regression Model


lin_reg = LinearRegression()

lin_reg = lin_reg.fit(X, y)

# Scores
print(model_scoring("Linear Regression", lin_reg, X, y))

## Decision tree Regresser


dec_tree = DecisionTreeRegressor()

dec_tree = dec_tree.fit(X, y)

# Scores
print(model_scoring("Decision Tree Regression", dec_tree, X, y))

## Support Vector Regressor


svr = SVR(kernel="linear")

y_ravel = y.ravel()

svr = svr.fit(X, y_ravel)

# Scores
print(model_scoring("Support Vector Regression", svr, X, y_ravel))

