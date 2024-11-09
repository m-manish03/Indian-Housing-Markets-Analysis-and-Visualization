import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from folium.plugins import MarkerCluster
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load Data
ban_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Bangalore.csv')
chn_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Chennai.csv')
del_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Delhi.csv')
hyd_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Hyderabad.csv')
mum_df = pd.read_csv('C:/Users/prave/Documents/MiniProject/Data/Mumbai.csv')

#Replacing the 9 values with NaN
ban_df.replace(9, np.nan, inplace=True)
chn_df.replace(9, np.nan, inplace=True)
del_df.replace(9, np.nan, inplace=True)
hyd_df.replace(9, np.nan, inplace=True)
mum_df.replace(9, np.nan, inplace=True)

#Converting the price from Rupees to Lakhs
ban_df['Price'] = ban_df['Price']/10**5
chn_df['Price'] = chn_df['Price']/10**5
del_df['Price'] = del_df['Price']/10**5
hyd_df['Price'] = hyd_df['Price']/10**5
mum_df['Price'] = mum_df['Price']/10**5

# Data Preparation and Visualization Functions
def merge_all(df1, df2, df3, df4, df5):
    df1['City'] = 'Bengaluru'
    df2['City'] = 'Chennai'
    df3['City'] = 'Delhi'
    df4['City'] = 'Hyderabad'
    df5['City'] = 'Mumbai'
    merged_df = pd.concat([df1, df2, df3, df4, df5]).reset_index(drop=True)
    return merged_df

merged_df = merge_all(ban_df, chn_df, del_df, hyd_df, mum_df)

def data_plots(df, city):
    st.write(f"### {city} Housing Data Analysis")
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

def merged_plots(merged_df):
    st.write("### Merged Dataset Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="City", y="Price", data=merged_df)
    st.pyplot(fig)
    
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

# Streamlit App
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["Home", "Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai"])

# Main Page
if selection == "Home":
    st.title("Indian Housing Market Analysis")
    st.write("## Project Description")
    st.write("Analysis of housing markets in major Indian cities with regression models.")
    
    merged_plots(merged_df)
    #st.write("### Model Performance on Data")
    #merged_scores = models_evaluation(merged_df)
    #st.dataframe(merged_scores)
    
    # Interactive Map
    st.sidebar.write("## Interactive Housing Map")
    df_map = pd.read_csv('Data/Combined_data.csv')
    city_map = folium.Map(location=[17.401, 78.477], zoom_start=12)
    mc = MarkerCluster()
    for idx, row in df_map.iterrows():
        mc.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']], popup=row['Location']))
    city_map.add_child(mc)
    folium_static(city_map)

# City Pages
else:
    st.title(f"{selection} Housing Market Analysis")
    city_dfs = {"bengaluru":ban_df,"chennai":chn_df,"delhi": del_df,"hyderabad":hyd_df,"mumbai_df":mum_df}
    city_data = city_dfs.get(selection.lower())
    data_plots(city_data, selection)
    #st.write(f"### Model Performance in {selection}")
    #city_scores = models_evaluation(city_data)
    #st.dataframe(city_scores)
