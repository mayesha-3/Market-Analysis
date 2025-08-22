import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules



df = pd.read_csv('CleanData.csv')
df['Revenue'] = df['Quantity'] * df['Price']
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df['Hour'] = df['Date'].dt.hour
df['Weekday'] = df['Date'].dt.day_name()

#Top selling by revenue and quantity
top_products = df.groupby('Itemname')[['Quantity', 'Revenue']].sum().sort_values(by='Revenue', ascending=False)

#top customers
top_customers = df.groupby('CustomerID')['Revenue'].sum().sort_values(ascending=False)

#best selling days and hours
daily_sales = df.groupby(df['Date'].dt.date)['Revenue'].sum()
hourly_sales = df.groupby('Hour')['Revenue'].sum()

#most sold categories

#Trends
monthly_sales = df.groupby('Month')['Revenue'].sum()
weekday_sales = df.groupby('Weekday')['Revenue'].sum()

#growth 
monthly_growth = monthly_sales.pct_change()


#seasonality
monthly_avg = monthly_sales.groupby(monthly_sales.index.month).mean()

#total spent
customer_metrics = df.groupby('CustomerID').agg({
    'Revenue': 'sum',
    'BillNo' : pd.Series.nunique,
    'Itemname' : pd.Series.nunique

}).rename(columns={'BillNo': 'PurchaseFrequency', 'Itemname': 'ProductDiversity'})

basket = df.groupby(['BillNo', 'Itemname'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)



frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

country_revenue = df.groupby('Country')['Revenue'].sum()
avg_transaction = df.groupby(['Country', 'BillNo'])['Revenue'].sum().groupby('Country').mean()
correlation = df[['Price', 'Quantity']].corr()
outliers = df[df['Quantity'] > df['Quantity'].quantile(0.99)]

#fraud
refunds = df[df['Quantity'] < 0]
high_value = df[df['Revenue'] > df['Revenue'].quantile(0.99)]


#repeat customer
repeat_customers = df.groupby('CustomerID')['BillNo'].nunique()
loyal = repeat_customers[repeat_customers > 1]

#Time gap
df_sorted = df.sort_values(by=['CustomerID', 'Date'])
df_sorted['PrevDate'] = df_sorted.groupby('CustomerID')['Date'].shift()
df_sorted['GapDays'] = (df_sorted['Date'] - df_sorted['PrevDate']).dt.days
df['MonthStr'] = df['Month'].astype(str)


st.title("Retail Data Dashboard")

st.sidebar.header("Filters")
selected_month = st.sidebar.selectbox("Select Month", df['MonthStr'].unique())
filtered_df = df[df['MonthStr'] == selected_month]


st.subheader("Top-Selling Products")
top_products = filtered_df.groupby('Itemname')[['Quantity', 'Revenue']].sum().sort_values(by='Revenue', ascending=False)
st.dataframe(top_products.head(10))

st.subheader("Monthly Sales Trend")
monthly_sales = df.groupby('Month')['Revenue'].sum()
st.line_chart(monthly_sales)

st.subheader("Basket Analysis Rules")
st.dataframe(rules[['antecedents', 'consequents', 'lift']])


if __name__ == "__main__":
    st.write("Dashboard loaded successfully.")