import pandas as pd
import datetime as dt

# Reading dataset
base_df = pd.read_csv('Datasets/data.csv')
df = base_df.copy()

# Inspecting the dataset
df.head(10)
df.info()
df.describe().T
df.isnull().sum()

# There are 2 columns for each order_num_total_ever and customer_value_total_ever. Creating single columns for each of them.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]
df["total_order"] = df["total_order"].astype('int64')

# We can see that there are Date columns. Changing column's type to datetime.
df.info()
date_columns = [col for col in df.columns if col.__contains__("date")]
for col in date_columns:
    df[col] = df[col].astype('datetime64[ns]')

# Checking out top 10 customers who spent more money than others.
df.groupby("master_id").agg({"total_value": "sum"}).sort_values("total_value", ascending=False).head(10)

# Checking out top 10 customers who made more purchases than others.
df.groupby("master_id").agg({"total_order": "sum"}).sort_values("total_order", ascending=False).head(10)

# Max date in the dataset is 30/05/2021. Setting analysis day +2.
today_date = dt.datetime(2021, 6, 1)

# Creating new RFM dataframe that includes every customer' Recency, Frequency and Monetary values.
rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                                   "total_order": lambda total_order: total_order.sum(),
                                   "total_value": lambda total_value: total_value.sum()})

# Renamed columns of new RFM dataframe.
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Creating scores of every RFM metrics between [1-5], 5 is best.
rfm['recency_score'] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1])
rfm['frequency_score'] = pd.qcut(rfm["Frequency"].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['monetary_score'] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5])

# Creating RFM scores of each customers.
rfm['RFM_SCORE'] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

# Creating segments for customers. "55 and 54" are best scores.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_lose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Creating segment column.
rfm["SEGMENT"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

# Analyzing segments for each RFM metric' mean values.
rfm.groupby("SEGMENT").agg({"Recency": "mean", "Frequency": "mean", "Monetary": "mean"})

# Sample Case: Creating new dataframe that includes customers from the [champions, loyal_customers] segment, who are interested in the “KADIN” category.
new_df = df[df["master_id"].isin(rfm[(rfm["SEGMENT"].isin(['champions', 'loyal_customers']))].index) & df["interested_in_categories_12"].apply(lambda x: " KADIN" in x)]["master_id"]
new_df.to_csv("SampleCaseOne.csv")


# Sample Case: Creating new dataframe that includes customers from the [cant_lose, hibernating, new_customers] segment, who are interested in "ERKEK" and "COCUK" categories.
new_df = df[df["master_id"].isin(rfm[(rfm['SEGMENT'].isin(["cant_lose", "hibernating", "new_customers"]))].index) & df['interested_in_categories_12'].apply(lambda x: " ERKEK" and " COCUK" in x)]["master_id"]
new_df.to_csv("SampleCaseTwo.csv")
