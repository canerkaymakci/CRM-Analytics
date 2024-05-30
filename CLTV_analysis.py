import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.float_format', lambda x: '%.4f' % x)

# Reading dataset.
main_df = pd.read_csv('Datasets/data.csv')
df = main_df.copy()

# Outlier Suppression methods.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Inspecting the dataset.
df.head()
df.info()
df.isnull().sum()
df.nunique()
df.shape
df.describe().T

# Outlier Suppression for necessary columns.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")
replace_with_thresholds(df, "customer_value_total_ever_offline")

# There are 2 columns for each order_num_total_ever and customer_value_total_ever. Creating single columns for each of them.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# To calculate CLTV, selecting datas with “total_order” value greater than 1.
df = df[df["total_order"] > 1]

# We can see that there are Date columns. Changing column's type to datetime.
df.info()
date_columns = [col for col in df.columns if col.__contains__("date")]
for col in date_columns:
    df[col] = df[col].astype('datetime64[ns]')

# Max date in the dataset is 30/05/2021. Setting analysis day +2.
today_date = dt.datetime(2021, 6, 1)


# Calculating every customer' Recency value.
df["recency_cltv_weekly"] = (df["last_order_date"] - df["first_order_date"]).dt.days

# Calculating every customer' Tenure value.
df["T_weekly"] = (today_date - df["first_order_date"]).dt.days

# Creating new dataframe that includes every customer' Recency, Tenure, Frequency and Monetary values.
new_df = df.groupby("master_id").agg({"recency_cltv_weekly": lambda recency: recency,
                                      "T_weekly": lambda t: t,
                                      "total_order": lambda total_order: total_order.astype(int),
                                      "total_value": lambda total_value: total_value})

# Renamed columns of new dataframe.
new_df.columns = ["recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]

# Recency and T values are weekly values. Editing values of every customer
new_df["recency_cltv_weekly"] = new_df["recency_cltv_weekly"] / 7
new_df["T_weekly"] = new_df["T_weekly"] / 7

# Monetary value is average monetary value.
new_df["monetary_cltv_avg"] = new_df["monetary_cltv_avg"] / new_df["frequency"]

# Creating BG/NBD model.
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Fitting the model.
bgf.fit(new_df["frequency"], new_df["recency_cltv_weekly"], new_df["T_weekly"])

# Sample predicts for expected sales.
new_df["exp_sales_3_months"] = bgf.predict(4*3, new_df["frequency"], new_df["recency_cltv_weekly"], new_df["T_weekly"])
new_df["exp_sales_6_months"] = bgf.predict(4*6, new_df["frequency"], new_df["recency_cltv_weekly"], new_df["T_weekly"])

# Creating Gamma-Gamma model.
ggf = GammaGammaFitter(penalizer_coef=0.001)

# Fitting Gamma-Gamma model.
ggf.fit(new_df["frequency"], new_df["monetary_cltv_avg"])

# Expected average values for every customer.
new_df["exp_average_value"] = ggf.conditional_expected_average_profit(new_df["frequency"], new_df["monetary_cltv_avg"])

# Calculating CLTV for each customer.
new_df["CLTV"] = ggf.customer_lifetime_value(bgf,
                                             new_df["frequency"],
                                             new_df["recency_cltv_weekly"],
                                             new_df["T_weekly"],
                                             new_df["monetary_cltv_avg"],
                                             time=6,
                                             freq='W')

# Creating segments for each customer based on CLTV.
new_df["SEGMENT"] = pd.qcut(new_df["CLTV"], 4, labels=["D", "C", "B", "A"])

# Analyzing segments.
new_df.groupby("SEGMENT").agg({"mean", "std", "count", "max", "min"})
