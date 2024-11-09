# %% [markdown]
# ### Getting started

# %%
# Data manipulation and analysis
import numpy as np
import pandas as pd

# Data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Machine learning models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# Preprocessing and model evaluation
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# %% [markdown]
# Loading all the datasets into respective dataframes

# %%
ban_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Bangalore.csv')
chn_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Chennai.csv')
del_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Delhi.csv')
hyd_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Hyderabad.csv')
mum_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Mumbai.csv')

# %% [markdown]
# Exploration of the dataframes.

# %%
hyd_df

# %% [markdown]
# Dataset Overview
# 
# The dataset in question is a collection of scraped data that encompasses a wide range of information related to the real estate market in India. Specifically, it includes a total of 40 explanatory variables that describe different aspects of houses in Indian metropolitan areas such as:

# %%
def print_column_table(df, columns_per_row=3):
    max_length = max(len(col) for col in df.columns)
    
    for i in range(0, len(df.columns), columns_per_row):
        row_columns = df.columns[i:i+columns_per_row]
        row = " ".join(f"{col:<{max_length}}" for col in row_columns)
        row = row.ljust(max_length * columns_per_row - 1) + " "
        print(row)

print_column_table(hyd_df, columns_per_row=6)

# %% [markdown]
# Important Note
# 
# It's essential to understand that for some houses, information about certain amenities was not provided. In such cases, the value '9' was used to indicate the absence of information about the apartment. However, it's crucial to recognize that these '9' values do not necessarily mean the absence of such a feature in real life.
# 
# '9' values do not represent actual data and could skew statistical analyses if treated as real values, leading to inaccurate conclusions.
# 
# To address the issue of missing data, all instances of '9' were replaced with NaN (Not a Number). This is a standard approach to represent missing data in pandas DataFrames, ensuring that missing data is properly handled and distinguished from actual data.
# 
# By replacing '9' with NaN, the analysis ensures that missing data is properly represented and can be handled appropriately in subsequent data cleaning, exploration, and modeling steps, ultimately leading to more accurate and reliable results.

# %%
#Replacing the 9 values with NaN
ban_df.replace(9, np.nan, inplace=True)
chn_df.replace(9, np.nan, inplace=True)
del_df.replace(9, np.nan, inplace=True)
hyd_df.replace(9, np.nan, inplace=True)
mum_df.replace(9, np.nan, inplace=True)

#Displaying the shape of the dataframes
print("Bengaluru: ", ban_df.shape)
print("Chennai: ", chn_df.shape)
print("Delhi: ", del_df.shape)
print("Hyderabad: ", hyd_df.shape)
print("Mumbai: ", mum_df.shape)

# %% [markdown]
# Lets Visualize these missing values

# %%
# Visualization using Matplotlib and Seaborn Libraries
fig, ax = plt.subplots(3, 2, figsize=(30,35), sharex=True)

# Bengaluru
sns.heatmap(ban_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[0][0])
ax[0][0].set_title('Bengaluru\nPrice: {} | Area: {} | Location: {}'.format(ban_df['Price'].mean(), ban_df['Area'].mean(), ban_df['Location'].nunique()))

# Chennai
sns.heatmap(chn_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[0][1])
ax[0][1].set_title('Chennai\nPrice: {} | Area: {} | Location: {}'.format(chn_df['Price'].mean(), chn_df['Area'].mean(), chn_df['Location'].nunique()))

# Delhi
sns.heatmap(del_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[1][0])
ax[1][0].set_title('Delhi\nPrice: {} | Area: {} | Location: {}'.format(del_df['Price'].mean(), del_df['Area'].mean(), del_df['Location'].nunique()))

# Hyderabad
sns.heatmap(hyd_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[1][1])
ax[1][1].set_title('Hyderabad\nPrice: {} | Area: {} | Location: {}'.format(hyd_df['Price'].mean(), hyd_df['Area'].mean(), hyd_df['Location'].nunique()))

# Mumbai
sns.heatmap(mum_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax[2][0])
ax[2][0].set_title('Mumbai\nPrice: {} | Area: {} | Location: {}'.format(mum_df['Price'].mean(), mum_df['Area'].mean(), mum_df['Location'].nunique()))

fig.delaxes(ax[2][1])

plt.show()

# %%
#Converting the price from Rupees to Lakhs

ban_df['Price'] = ban_df['Price']/10**5
chn_df['Price'] = chn_df['Price']/10**5
del_df['Price'] = del_df['Price']/10**5
hyd_df['Price'] = hyd_df['Price']/10**5
mum_df['Price'] = mum_df['Price']/10**5

# %% [markdown]
# Combining all cities datasets

# %%
def merge_all(df1, df2, df3, df4, df5):
    df1['City'] = 'Bengaluru'
    df2['City'] = 'Chennai'
    df3['City'] = 'Delhi'
    df4['City'] = 'Hyderabad'
    df5['City'] = 'Mumbai'
    merged_df = pd.concat([ban_df, chn_df, del_df, hyd_df, mum_df]).reset_index(drop=True)
    return merged_df

# %% [markdown]
# #### EDA Functions
# 
# First the different functions for Exploratory Data Analysis are created

# %%
#Function to visualize the necessary data of each city.
def data_plots(df, city):
    #Scatterplot of the Area and price against no of bedrooms in each city
    plt.figure(figsize=(15, 8))
    sns.scatterplot(x=df['Price'], y=df['Area'], hue=df['No. of Bedrooms'])
    plt.title('Area(sq. ft) vs Price(lakhs) in '+city)
    plt.show()

    #Countplot of the Properties at every location in each city
    plt.figure(figsize=(20, 8))
    sns.countplot(y='Location', data=df, order=df.Location.value_counts().index[:25])
    plt.title('Number of units at each location in '+city)
    plt.show()
    
    # Bar plot of Affordable locations by location
    plt.figure(figsize=(10, 10))
    most_price_by_location = df.groupby('Location')['Price'].mean().sort_values().head(40)
    sns.barplot(x=most_price_by_location.values, y=most_price_by_location.index)
    plt.title('Most Affordable Areas in ' + city)
    plt.show()
    
    # Bar plot for Most Demanded Locations sorted by average area
    plt.figure(figsize=(10, 10))
    most_demanded_locations = df.groupby('Location').agg({'Area': 'mean', 'Location': 'count'}).rename(columns={'Location': 'Count'})
    most_demanded_locations = most_demanded_locations.sort_values(by='Area', ascending=False).head(40)
    sns.barplot(x=most_demanded_locations['Count'], y=most_demanded_locations.index, palette='viridis')
    plt.title('Most Demanded Areas in ' + city)
    plt.show()
    
    #Distribution plot of Sale Price across each city
    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    fig.suptitle('Distribution of Sales across '+city)
    sns.histplot(df['Price'], kde=True, ax=ax[0])
    sns.histplot(np.log(df['Price']), kde=True, ax=ax[1])
    plt.show()
    
    #Catplots of Price and Area against each City in the Merged data
    sns.catplot(y='Location', x='Price', data=df, jitter=0.15, height=10, aspect=2)
    plt.title('House Price(in lakhs) variation in these cities')
    plt.show()
    sns.catplot(y='Location', x='Area', data=df, jitter=0.15, height=10, aspect=2)
    plt.title('House Area(in sq. ft) variation in these cities')
    plt.show()

    #Correlation heatmap of the attributes in each city
    numeric_correlation = df.select_dtypes(exclude=object).corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(numeric_correlation, annot = True, square=True, fmt='.1f')
    plt.title('Correlation', fontsize=50)
    plt.show()

# %%
#Function to visualize the necessary data of all the cities merged.
def merged_plots(merged_df):
    #Catplots of Price and Area against each City in the Merged data
    sns.catplot(x='City', y='Price', data=merged_df, jitter=0.15, height=10, aspect=2)
    plt.title('House Price(in lakhs) variation in these cities')
    plt.show()
    sns.catplot(x='City', y='Area', data=merged_df, jitter=0.15, height=10, aspect=2)
    plt.title('House Area(in sq. ft) variation in these cities')
    plt.show()

    # Smooth line graph of the House price vs Area with respect to each city in the merged data
    #filtered_df = merged_df[merged_df['Price'] <= 600]
    window_size = 15
    smoothed_df = (
        merged_df.groupby('City')
        .apply(lambda x: x[['Price', 'Area']].rolling(window=window_size).mean())
        .reset_index()
    )
    smoothed_df['City'] = merged_df['City']
    plt.figure(figsize=(20,10))
    sns.lineplot(x=smoothed_df['Price'], y=smoothed_df['Area'], hue=smoothed_df['City'], linewidth=1)
    plt.title('House Price(lakhs) vs Area(sq. ft) across cities')
    #plt.gca().set_xticks(range(0, 1500, 100))
    plt.show()
    
    #Correlation heatmap of the Merged Data
    numeric_correlation = merged_df.select_dtypes(exclude=object).corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(numeric_correlation, annot = True, square=True, fmt='.1f')
    plt.title('Correlation', fontsize=50)
    plt.show()

# %% [markdown]
# #### machine learning models
# 
# The following functions are defined for analyzing and visualizing the performance of different machine learning models, including their scoring plots.
# 

# %%
#Function to implement these 6 regression models and evaluate their performance
def models_evaluation(df):
        #Encoding the Categorical Values
        label_encoder = LabelEncoder()
        df['Location'] = label_encoder.fit_transform(df.Location)

        #Creating the training data and test data
        X = df.drop(['Price'], axis=1)
        #To create higher efficiency we take the log value
        y = np.log(df['Price'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        #Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_y_pred = lr_model.predict(X_test)
        lr_score = lr_model.score(X_test, y_test)
        lr_cv = cross_val_score(estimator = lr_model, X = X_train, y = y_train, cv = 10)
        lr_mae = metrics.mean_absolute_error(y_test, lr_y_pred)
        lr_mse = metrics.mean_squared_error(y_test, lr_y_pred)
        lr_rmse = np.sqrt(lr_mse)
        lr_r2 = metrics.r2_score(y_test, lr_y_pred)

        # Polynomial Regression
        poly = PolynomialFeatures(degree=2, include_bias=False)
        scaler = StandardScaler()
        ridge = Ridge(alpha=1.0)  # You can adjust the alpha value
        pr_model = make_pipeline(poly, scaler, ridge)
        pr_model.fit(X_train, y_train)
        pr_y_pred = pr_model.predict(X_test)
        pr_score = pr_model.score(X_test, y_test)
        pr_cv = cross_val_score(estimator=pr_model, X=X_train, y=y_train, cv=10)
        pr_mae = metrics.mean_absolute_error(y_test, pr_y_pred)
        pr_mse = metrics.mean_squared_error(y_test, pr_y_pred)
        pr_rmse = np.sqrt(pr_mse)
        pr_r2 = metrics.r2_score(y_test, pr_y_pred)

        #Decision Tree
        dt_model = DecisionTreeRegressor(random_state=0)
        dt_model.fit(X_train, y_train)
        dt_y_pred = dt_model.predict(X_test)
        dt_score = dt_model.score(X_test, y_test)
        dt_cv = cross_val_score(estimator = dt_model, X = X_train, y = y_train, cv = 10)
        dt_mae = metrics.mean_absolute_error(y_test, dt_y_pred)
        dt_mse = metrics.mean_squared_error(y_test, dt_y_pred)
        dt_rmse = np.sqrt(dt_mse)
        dt_r2 = metrics.r2_score(y_test, dt_y_pred)

        #Random Forest
        rf_model = RandomForestRegressor(n_estimators=500, random_state=0)
        rf_model.fit(X_train, y_train)
        rf_y_pred = rf_model.predict(X_test)
        rf_score = rf_model.score(X_test, y_test)
        rf_cv = cross_val_score(estimator = rf_model, X = X_train, y = y_train, cv = 10)
        rf_mae = metrics.mean_absolute_error(y_test, rf_y_pred)
        rf_mse = metrics.mean_squared_error(y_test, rf_y_pred)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = metrics.r2_score(y_test, rf_y_pred)

        #Suport Vector Machine(Support Vector Regression)
        sc = StandardScaler()
        X_train_scaled = sc.fit_transform(X_train)
        X_test_scaled = sc.fit_transform(X_test)
        svr_model = SVR(kernel='rbf')
        svr_model.fit(X_train_scaled, y_train)
        svr_y_pred = svr_model.predict(sc.transform(X_test))
        svr_score = svr_model.score(X_test_scaled, y_test)
        svr_cv = cross_val_score(estimator = svr_model, X = X_train_scaled, y = y_train, cv = 10)
        svr_mae = metrics.mean_absolute_error(y_test, svr_y_pred)
        svr_mse = metrics.mean_squared_error(y_test, svr_y_pred)
        svr_rmse = np.sqrt(svr_mse)
        svr_r2 = metrics.r2_score(y_test, svr_y_pred)

        #K-Nearest Neighbors
        knn_model = KNeighborsRegressor(n_neighbors=5)  # adjust the number of neighbors
        knn_model.fit(X_train, y_train)
        knn_y_pred = knn_model.predict(X_test)
        knn_score = knn_model.score(X_test, y_test)
        knn_cv = cross_val_score(estimator=knn_model, X=X_train, y=y_train, cv=10)
        knn_mae = metrics.mean_absolute_error(y_test, knn_y_pred)
        knn_mse = metrics.mean_squared_error(y_test, knn_y_pred)
        knn_rmse = np.sqrt(knn_mse)
        knn_r2 = metrics.r2_score(y_test, knn_y_pred)

        #Extreme Gradient Boosting
        xgb_model = XGBRegressor()
        xgb_model.fit(X_train, y_train)
        xgb_y_pred = xgb_model.predict(X_test)
        xgb_score = xgb_model.score(X_test, y_test)
        xgb_cv = cross_val_score(estimator = xgb_model, X = X_train, y = y_train, cv = 10)
        xgb_mae = metrics.mean_absolute_error(y_test, xgb_y_pred)
        xgb_mse = metrics.mean_squared_error(y_test, xgb_y_pred)
        xgb_rmse = np.sqrt(xgb_mse)
        xgb_r2 = metrics.r2_score(y_test, xgb_y_pred)

        #Store the metrics result of each model in a list return it
        mods = [('Linear Regression', lr_r2, lr_cv.mean(), lr_mae, lr_mse, lr_rmse),
                ('Polynomial Regression(^2)', pr_r2, pr_cv.mean(), pr_mae, pr_mse, pr_rmse),
                ('Decision Tree Regression', dt_r2, dt_cv.mean(), dt_mae, dt_mse, dt_rmse),
                ('Random Forest Regression', rf_r2, rf_cv.mean(), rf_mae, rf_mse, rf_rmse),
                ('Support Vector Regression', svr_r2, svr_cv.mean(), svr_mae, svr_mse, svr_rmse),
                ('K-Nearest Neighbors', knn_r2, knn_cv.mean(), knn_mae, knn_mse, knn_rmse),
                ('XGBoost Regression', xgb_r2, xgb_cv.mean(), xgb_mae, xgb_mse, xgb_rmse)
        ]
        scores = pd.DataFrame(data=mods, columns=['Model', 'R2_Score', 'Cross-Validation', 'MAE', 'MSE', 'RMSE'])

        return scores

# %%
#Function to plot the 5 performance measures of the models
def score_plots(df, city):
    #Barplots in subplots to plot the performance
    fig, ax = plt.subplots(3,2, figsize=(35,20))

    fig.suptitle(city, fontsize=50)

    sns.barplot(y='Model', x='R2_Score', data=df, ax=ax[0][0])

    df.sort_values(by='Cross-Validation', inplace=True)
    sns.barplot(y='Model', x='Cross-Validation', data=df, ax=ax[0][1])
    
    df.sort_values(by='MAE', inplace=True)
    sns.barplot(y='Model', x='MAE', data=df, ax=ax[1][0])
    
    df.sort_values(by='MSE', inplace=True)
    sns.barplot(y='Model', x='MSE', data=df, ax=ax[1][1])
    
    df.sort_values(by='RMSE', inplace=True)
    sns.barplot(y='Model', x='RMSE', data=df, ax=ax[2][0])
    
    fig.delaxes(ax[2][1])

    plt.show()

# %% [markdown]
# 
# 
# #### Proceeding with dropping the missing values

# %%
#Dropping the missing values
ban_df = ban_df.dropna()
chn_df = chn_df.dropna()
del_df = del_df.dropna()
hyd_df = hyd_df.dropna()
mum_df = mum_df.dropna()

#Checking the data size
print("Bengaluru: ", ban_df.shape)
print("Chennai: ", chn_df.shape)
print("Delhi: ", del_df.shape)
print("Hyderabad: ", hyd_df.shape)
print("Mumbai: ", mum_df.shape)

# %% [markdown]
# ## Exploratory Data Analysis of each dataset and the merged one followed by the model training and performance results!

# %% [markdown]
# #### Bengaluru
# EDA:

# %%
data_plots(ban_df, 'Bengaluru')

# %% [markdown]
# Model Training and Performance:

# %%
bang_scores = models_evaluation(ban_df)
bang_scores

# %%
score_plots(bang_scores, 'Bengaluru')

# %% [markdown]
# #### Chennai
# EDA:

# %%
data_plots(chn_df,'Chennai')

# %% [markdown]
# Model Training and Performance:

# %%
chn_scores = models_evaluation(chn_df)
chn_scores

# %%
score_plots(chn_scores, 'Chennai')

# %% [markdown]
# #### Delhi

# %% [markdown]
# EDA:

# %%
data_plots(del_df, 'Delhi')

# %% [markdown]
# Model Training and Performance:

# %%
del_scores = models_evaluation(del_df)
del_scores

# %%
score_plots(del_scores, 'Delhi')

# %% [markdown]
# #### Hyderabad
# EDA:

# %%
data_plots(hyd_df, 'Hyderabad')

# %% [markdown]
# Model Training and Performance:

# %%
hyd_scores = models_evaluation(hyd_df)
hyd_scores

# %%
score_plots(hyd_scores, 'Hyderabad')

# %% [markdown]
# #### Mumbai
# EDA:

# %%
data_plots(mum_df, 'Mumbai')

# %% [markdown]
# Model Training and Performance:

# %%
mum_scores = models_evaluation(mum_df)
mum_scores

# %%
score_plots(mum_scores, 'Mumbai')

# %% [markdown]
# #### Merged Dataset

# %%
merged_df = merge_all(ban_df, chn_df, del_df, hyd_df, mum_df)

# %% [markdown]
# EDA:

# %%
merged_plots(merged_df)

# %% [markdown]
# Model Training and Performance:

# %%
merged_df['City'] = LabelEncoder().fit_transform(merged_df.City)
mergeddf_scores = models_evaluation(merged_df)
mergeddf_scores

# %%
score_plots(mergeddf_scores, 'Merged')

# %% [markdown]
# # Interactive Housing Market Map
# An interactive map visualization of housing locations across major Indian cities, the map provides a comprehensive view of the real estate landscape, allowing users to explore property locations visually

# %%
import pandas as pd
import folium
from folium.plugins import MarkerCluster

df = pd.read_csv('C:/Users/prave/Documents/GitHub/Housing market/MiniProject/Data/Combined_data.csv')

# Create a map centered on Hyderabad
city_map = folium.Map(location=[17.401, 78.477], zoom_start=15, tiles='cartodbpositron')

# Create a MarkerCluster
mc = MarkerCluster()

# Add markers for each location
for idx, row in df.iterrows():
    popup_text = row['Location']
    mc.add_child(folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_text
    ))

# Add the MarkerCluster to the map
city_map.add_child(mc)

# Save the map to an HTML file
city_map.save("housing_map.html")

city_map