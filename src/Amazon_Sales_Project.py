# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
sns.set_context("notebook")


# %%
data = pd.read_csv(
    "C:\\Users\\lenovo\\OneDrive\\Desktop\\CV-PROJECTS\\Amazon Sales Dataset.zip",
    compression='zip')

# %%
data.head()

# %%
data.shape

# %%
data.info()

# %%
data.describe()

# %%
data.duplicated().sum()

# %%
for col in data.columns:
    if data[col].nunique() < data.shape[0]:
        print(col)

# %%
data.isnull().sum()

# %%
# %%
# Fixigrating_count column
data['rating_count'] = (
    data['rating_count']
    .str.replace(',', '', regex=False)
    .astype(float)
)

# missing values
data['rating_count'].fillna(data['rating_count'].median(), inplace=True)


# %%
data.isnull().sum()



#DATA CLEANING
# %%
data.drop_duplicates(inplace=True)

data['actual_price'] = (data['actual_price'].str.replace('₹','', regex=False).str.replace(',','', regex=False).astype(float))

data['discounted_price'] = (data['discounted_price'].str.replace('₹','', regex=False).str.replace(',','', regex=False).astype(float))

data['discount_percentage'] = (data['discount_percentage'].str.replace('%','', regex=False).astype(float))


# %%
data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
data['rating'].fillna(data['rating'].median(), inplace=True)


#CATEGORY AD DISCOUNT
# %%
data['category'].value_counts().head(10)

# %%
data.sort_values('discount_percentage', ascending=False).head(10)


#UNIVARIATE ANALYSIS
#ACTUAL PRICE
# %%
plt.figure(figsize=(8,5))
sns.histplot(data['actual_price'], bins=30)
#plt.xscale('log')
plt.title("Actual Price")
plt.show()


# DISCOUNT PERCENTAGE
# %%
plt.figure(figsize=(8,5))
sns.histplot(data['discount_percentage'], bins=20)
plt.title("Discount Percentage")
plt.show()

#PRODUCT RATING
# %%
plt.figure(figsize=(8,5))
sns.histplot(data['rating'], bins=10, kde=True)
plt.title("Rating PLOT")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()



# %%
#Category-wise products sales
plt.figure(figsize=(10,5))
top_categories = data['category'].value_counts().head(10)

sns.barplot(
    x=top_categories.values,
    y=top_categories.index)
plt.title("Top 10 Product Categories")
plt.xlabel("Number of Products")
plt.ylabel("Category")
plt.show()



#BIVARIATE ANALYSIS

#DISCOUNT vs RATING
# %%
plt.figure(figsize=(8,5))
sns.regplot(
    x='discount_percentage',
    y='rating',
    data=data,
    scatter_kws={'alpha':0.4}
)
plt.title("Discount vs Rating")
plt.show()


#ACTUAL vs DISCOUNTED PRICE
# %%
plt.figure(figsize=(8,5))
sns.regplot(
    x='actual_price',
    y='discounted_price',
    data=data,
    scatter_kws={'alpha':0.4}
)
plt.xscale('log')
plt.yscale('log')
plt.title("Actual Price vs Discounted Price")
plt.show()


#OUTLIER DETECTION
# %%
plt.figure(figsize=(8,5))
sns.boxplot(x=np.log10(data['actual_price']))
plt.title("Outliers in Actual Price (Log Scale)")
plt.show()

# %%
#HEATMAP PLOT
sns.heatmap(
    data[['actual_price','discounted_price','discount_percentage','rating']].corr(),
    annot=True
)
plt.show()

# %%
