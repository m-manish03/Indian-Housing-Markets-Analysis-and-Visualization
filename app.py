import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Load Data
ban_df = pd.read_csv('Data/Bangalore.csv')
chn_df = pd.read_csv('Data/Chennai.csv')
del_df = pd.read_csv('Data/Delhi.csv')
hyd_df = pd.read_csv('Data/Hyderabad.csv')
mum_df = pd.read_csv('Data/Mumbai.csv')

#Replacing the 9 values with NaN
ban_df.replace(9, np.nan, inplace=True)
chn_df.replace(9, np.nan, inplace=True)
del_df.replace(9, np.nan, inplace=True)
hyd_df.replace(9, np.nan, inplace=True)
mum_df.replace(9, np.nan, inplace=True)

#Dropping the missing values
ban_df = ban_df.dropna()
chn_df = chn_df.dropna()
del_df = del_df.dropna()
hyd_df = hyd_df.dropna()
mum_df = mum_df.dropna()

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
    # Top 10 Popular Locations by Count
    st.write("### Top 10 Popular Locations ")
    top_locations = df['Location'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_locations.values, y=top_locations.index, palette='Blues_r', ax=ax)
    ax.set_title('Top 10 Popular Locations')
    ax.set_xlabel('Number of Listings')
    st.pyplot(fig)

    # Bar plot of Affordable locations by location
    st.write('### Most Affordable Areas by Median Price in ' + city)
    fig, ax = plt.subplots(figsize=(10, 10))
    most_price_by_location = df.groupby('Location')['Price'].mean().sort_values().head(20)
    sns.barplot(x=most_price_by_location.values, y=most_price_by_location.index, ax=ax)
    ax.set_title('Most Affordable Areas by Median Price in ' + city)
    st.pyplot(fig)

    # Median Price per Neighborhood
    st.write("### Most Expensive Areas by Median Price in " + city)
    top_neighborhoods = df.groupby(['City', 'Location'])['Price'].median().reset_index()
    top_neighborhoods = top_neighborhoods.groupby('City').apply(lambda x: x.nlargest(20, 'Price')).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=top_neighborhoods, y="Location", x="Price", hue="City", ax=ax)
    ax.set_title("Most Expensive Areas by Median Price in " + city)
    st.pyplot(fig)
    
    # Bar plot for Most Demanded Locations sorted by average area
    st.write('### Most Demanded Locations in ' + city)
    fig, ax = plt.subplots(figsize=(10, 10))
    most_demanded_locations = df.groupby('Location').agg({'Price': 'mean', 'Area': 'mean', 'Location': 'count'}).rename(columns={'Location': 'Count'})
    most_demanded_locations = most_demanded_locations.sort_values(by=['Count', 'Area', 'Price'], ascending=False).head(40)
    sns.barplot(x=most_demanded_locations['Count'], y=most_demanded_locations.index, palette='viridis', ax=ax)
    ax.set_title('Most Demanded Locations in ' + city)
    st.pyplot(fig)
    
    # Countplot of the Properties at every location in each city
    st.write('### Number of units at each location ' + city)
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.countplot(y='Location', data=df, order=df.Location.value_counts().index[:25], ax=ax)
    ax.set_title('Number of units at each location in ' + city)
    st.pyplot(fig) 

    #Distribution plot of Sale Price across each city
    st.write('### Distribution of Sales across ' + city)
    fig, ax = plt.subplots(1, 2, figsize=(20,5))
    fig.suptitle('Distribution of Sales across '+city)
    sns.histplot(df['Price'], kde=True, ax=ax[0])
    sns.histplot(np.log(df['Price']), kde=True, ax=ax[1])
    st.pyplot(fig)
    
    # Scatter plot
    st.write('### Area(sq. ft) vs Price(lakhs) in ' + city)
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.scatterplot(x=df['Price'], y=df['Area'], hue=df['No. of Bedrooms'], ax=ax)
    ax.set_title('Area(sq. ft) vs Price(lakhs) in ' + city)
    st.pyplot(fig)
    
    # Price vs Area Scatter Plot with Trend Line
    st.write("### Price vs Area Scatter Plot with Trend Line")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=df, x='Area', y='Price', hue='City', palette='tab10', ax=ax)
    sns.regplot(data=df, x='Area', y='Price', scatter=False, color='gray', line_kws={"linewidth": 1, "linestyle": "dashed"}, ax=ax)
    ax.set_title('Price vs Area with Trend Line by City')
    ax.set_xlabel('Area (sq. ft)')
    ax.set_ylabel('Price (in lakhs)')
    st.pyplot(fig)
    
    #Catplots of Price and Area
    st.write('### House Price(in lakhs) variation in ' + city)
    fig1 = sns.catplot(y='Location', x='Price', data=df, jitter=0.15, height=15, aspect=0.5)
    fig1.figure.suptitle('House Price(in lakhs) variation in ' + city)
    st.pyplot(fig1.figure)
    
    st.write('###  House Area(in sq. ft) variation in ' + city)
    fig2 = sns.catplot(y='Location', x='Area', data=df, jitter=0.15, height=15, aspect=0.5)
    fig2.figure.suptitle('House Area(in sq. ft) variation in ' + city)
    st.pyplot(fig2.figure)
    
    if view_mode == "Analyst":
        #Correlation heatmap of the attributes in each city
        st.write('### Correlation heatmap of the attributes in ' + city)
        numeric_correlation = df.select_dtypes(exclude=object).corr()
        fig3, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(numeric_correlation, annot=True, square=True, fmt='.1f', ax=ax)
        ax.set_title('Correlation', fontsize=50)
        st.pyplot(fig3)
    
def merged_plots(merged_df,combined_df):
    '''
    st.write("### Merged Dataset Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="City", y="Price", data=merged_df)
    st.pyplot(fig)
    '''
    # Heatmap of Most Expensive Locations
    st.write("### Heatmap of Most Expensive Locations")
    city_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center on India
    heat_data = [[row['Latitude'], row['Longitude'], row['Price']] for index, row in combined_df.iterrows() if pd.notnull(row['Latitude']) and pd.notnull(row['Longitude'])]
    HeatMap(heat_data, radius=10, max_zoom=13).add_to(city_map)
    folium_static(city_map)
    
    # Median Price per Neighborhood
    st.write("### Top Expensive Neighborhoods by Median Price")
    top_neighborhoods = merged_df.groupby(['City', 'Location'])['Price'].median().reset_index()
    top_neighborhoods = top_neighborhoods.groupby('City').apply(lambda x: x.nlargest(10, 'Price')).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(data=top_neighborhoods, y="Location", x="Price", hue="City", ax=ax)
    ax.set_title("Top Expensive Neighborhoods by Median Price")
    st.pyplot(fig)
    
    # Top 10 Popular Locations
    st.write("### Top 10 Popular Locations ")
    top_locations = merged_df['Location'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top_locations.values, y=top_locations.index, palette='Blues_r', ax=ax)
    ax.set_title('Top 10 Popular Locations')
    ax.set_xlabel('Number of Listings')
    st.pyplot(fig)
    
    #Catplots of Price and Area against each City in the Merged data
    st.write("### House Price variation in these cities")
    fig1 = sns.catplot(y='City', x='Price', data=merged_df, jitter=0.15, height=10, aspect=2)
    fig1.figure.suptitle('House Price(in lakhs) variation in these cities') 
    st.pyplot(fig1.figure)

    st.write("### House Area variation in these cities")
    fig2 = sns.catplot(y='City', x='Area', data=merged_df, jitter=0.15, height=10, aspect=2)
    fig2.figure.suptitle('House Area(in sq. ft) variation in these cities')
    st.pyplot(fig2.figure)
    
    # Price vs Area Scatter Plot with Trend Line
    st.write("### Price vs Area Scatter Plot with Trend Line by City")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=merged_df, x='Area', y='Price', hue='City', palette='tab10', ax=ax)
    sns.regplot(data=merged_df, x='Area', y='Price', scatter=False, color='gray', line_kws={"linewidth": 1, "linestyle": "dashed"}, ax=ax)
    ax.set_title('Price vs Area with Trend Line by City')
    ax.set_xlabel('Area (sq. ft)')
    ax.set_ylabel('Price (in lakhs)')
    st.pyplot(fig)
    
    # Smooth line graph of the House price vs Area with respect to each city in the merged data
    #filtered_df = merged_df[merged_df['Price'] <= 600]
    window_size = 15
    smoothed_df = (
        merged_df.groupby('City')
        .apply(lambda x: x[['Price', 'Area']].rolling(window=window_size).mean())
        .reset_index()
    )
    smoothed_df['City'] = merged_df['City']
    fig, ax = plt.subplots(figsize=(20,10))
    sns.lineplot(x=smoothed_df['Price'], y=smoothed_df['Area'], hue=smoothed_df['City'], linewidth=1,ax=ax)
    plt.title('House Price(lakhs) vs Area(sq. ft) across cities')
    #plt.gca().set_xticks(range(0, 1500, 100))
    st.pyplot(fig)
    
    # Interactive Map
    city_map = folium.Map(location=[17.401, 78.477], zoom_start=12)
    mc = MarkerCluster()
    for idx, row in combined_df.iterrows():
        mc.add_child(folium.Marker(location=[row['Latitude'], row['Longitude']], popup=row['Location']))
    city_map.add_child(mc)
    folium_static(city_map)
    
    if view_mode == "Analyst":
        #Correlation heatmap of the Merged Data
        numeric_correlation = merged_df.select_dtypes(exclude=object).corr()
        fig3, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(numeric_correlation, annot=True, square=True, fmt='.1f', ax=ax)
        ax.set_title('Correlation', fontsize=50)
        st.pyplot(fig3)

#Function to implement these 6 regression models and evaluate their performance
def models_evaluation(df):
        #Encoding the Categorical Values
        label_encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.fit_transform(df[col])

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

# Css
st.markdown("""
    <style>
    /* Style for sidebar */
    .css-1d391kg {  /* Sidebar class */
        background-color: #d0e7ff !important;  /* Light blue background */
    }
    /* Button styling */
    .stButton button {
        width: 100%;                /* Full width for each button */
        height: 50px;               /* Set a consistent height */
        font-size: 18px;            /* Font size for readability */
        margin-bottom: 10px;        /* Spacing between buttons */
        background-color: #1f77b4;  /* Button color */
        color: white;               /* Button text color */
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    selection = "Home"
elif st.sidebar.button("Bengaluru"):
    selection = "Bengaluru"
elif st.sidebar.button("Chennai"):
    selection = "Chennai"
elif st.sidebar.button("Delhi"):
    selection = "Delhi"
elif st.sidebar.button("Hyderabad"):
    selection = "Hyderabad"
elif st.sidebar.button("Mumbai"):
    selection = "Mumbai"
else:
    selection = "Home"
# Analyst toggle
view_mode = st.sidebar.selectbox("Select View", ["Simple", "Analyst"])

all_scores = pd.read_csv('Data/model_evaluation_results.csv')

# Main Page
if selection == "Home":
    st.title("Indian Housing Market Analysis and Visualization")
    st.markdown("""
    ### Introduction
    Welcome to the Indian Housing Markets Analysis and Visualization app. This tool provides a comprehensive analysis of the housing markets across major Indian cities, enabling users to explore trends, pricing dynamics, and demand hotspots in an interactive way.
    
    ### Project Description
    The app leverages advanced data analytics to uncover valuable insights from housing market data across key Indian cities. Using data visualization techniques, we present clear, data-driven insights into pricing trends, neighborhood demand, and affordability.
    
    ### Key Features
    - **City-wise Insights**: Analyze housing trends, average prices, and demand in major metropolitan areas.
    - **Interactive Visualizations**: Explore data with scatter plots, bar charts, count plots, and heatmaps to understand market trends.
    - **Location Analysis**: Identify affordable areas, high-demand neighborhoods, and view data on interactive maps.
        """)
    combined_df = pd.read_csv('Data/Combined_data.csv')
    merged_plots(merged_df,combined_df)
    if view_mode == "Analyst":
        st.write("### Model Performance on Data")
        #merged_scores = models_evaluation(merged_df)
        merged_scores = all_scores[all_scores['City'] == 'merged_df'].drop(columns=['City'])
        st.dataframe(merged_scores)
    
    
# City Pages
else:
    st.title(f"Housing Market Analysis of {selection}")
    city_dfs = {"bengaluru":ban_df,"chennai":chn_df,"delhi": del_df,"hyderabad":hyd_df,"mumbai_df":mum_df}
    city_data = city_dfs.get(selection.lower())
    data_plots(city_data, selection)
    if view_mode == "Analyst":
        st.write(f"### Model Performance in {selection}")
        #city_scores = models_evaluation(city_data)
        city_scores = all_scores[all_scores['City'] == selection].drop(columns=['City'])
        st.dataframe(city_scores)
