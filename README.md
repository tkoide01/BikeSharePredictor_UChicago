# BikeShare Rider Predictor: Unleash the Power of Machine Learning for Accurate Ridership Forecasts

## Description:
Welcome to BikeShare Rider Predictor, an innovative machine learning project that leverages the rich historical trip data of the Capital Bike Sharing Service to accurately forecast future ridership trends. Whether you're a data enthusiast, a bike-sharing operator, or a city planner, this GitHub repository is your gateway to unlocking valuable insights and making informed decisions.

By analyzing extensive data spanning over a decade, from 2010 to 2023, this state-of-the-art machine learning model has been trained to capture the intricate patterns and factors that influence bike-share ridership. Through careful feature engineering and advanced predictive algorithms, the model offers a reliable and efficient solution for predicting future ridership on a daily, weekly, and even monthly basis.

----------------------------------------------------------------------------------------------------------------------
## What is Capital Bike Share Service?
Capital Bike Share is a bike-sharing service that operates in the Washington, D.C. metropolitan area, including Washington, D.C., Arlington, Virginia, and Alexandria, Virginia. It provides a convenient and sustainable transportation option for residents and visitors, allowing them to rent bicycles for short trips.

The service offers a network of bike stations strategically located throughout the region, allowing users to easily pick up and drop off bikes at their desired locations. Riders can access bikes through a membership or by purchasing a short-term pass, making it accessible to both frequent and occasional users.

Capital Bike Share promotes active transportation, reducing reliance on cars and contributing to improved air quality and reduced traffic congestion. It offers a flexible and eco-friendly way to navigate the city, whether for commuting, leisurely rides, or running errands.

With a growing number of users, the service generates a wealth of data related to bike trips, including duration, start and end stations, member type, and timestamps. This data can be leveraged to gain insights into rider behavior, preferences, and patterns.

By analyzing and modeling this data using machine learning and predictive algorithms, it becomes possible to make accurate predictions about total trip counts, understand factors influencing bike usage, and optimize the availability and distribution of bikes across the network.

Capital Bike Share has become an integral part of the transportation ecosystem in the Washington, D.C. area, providing an accessible, sustainable, and efficient means of travel for individuals and contributing to a greener and more connected community. Currently, the service reports that it has over 30,000 active members and serves 3million+ of riders annually.

----------------------------------------------------------------------------------------------------------------------
## Key Features:

Comprehensive Dataset: Access an extensive and preprocessed dataset, covering millions of bike-sharing trips, collected over 13 years, to train and evaluate your own models.

Robust Machine Learning Model: Take advantage of our powerful machine learning model, carefully developed and fine-tuned to accurately predict bike-share ridership based on historical data. Save time and resources by building upon our well-established foundation.

Interactive Jupyter Notebooks: Explore and learn from a collection of Jupyter notebooks, providing step-by-step guides, explanations, and visualizations of the data preprocessing, model training, and evaluation process. Easily adapt these notebooks to your own project requirements.

Evaluation Metrics: Evaluate the performance of your own models with a set of commonly used evaluation metrics, including root mean squared error (RMSE), mean absolute error (MAE), and R-squared (R²). Compare your results with our benchmark model and strive for excellence.

Documentation and Best Practices: Benefit from comprehensive documentation, offering insights into data preprocessing techniques, feature engineering strategies, model selection considerations, and practical tips for improving model performance. Stay updated with the latest best practices in the field.

Unlock the potential of data-driven decision-making and optimize the operations of your bike-sharing service with BikeShare Rider Predictor. Access the repository now and embark on a journey towards accurate and insightful ridership forecasts. Let the power of machine learning revolutionize the way you understand and optimize urban mobility.

----------------------------------------------------------------------------------------------------------------------
## Data Description:

Data Source: I collected Capital Bikeshare trip data from its official website. 
URL: <https://s3.amazonaws.com/capitalbikeshare-data/index.html>

Data Description: Based on the information provided by the website, the Capital Bikeshare publish downloadable files of Capital Bikeshare trip data. Total of 34,500,331 data includes:

- Duration – Duration of trip
- Start Date – Includes start date and time
- End Date – Includes end date and time
- Start Station – Includes starting station name and number
- End Station – Includes ending station name and number
- Bike Number – Includes ID number of bike used for the trip
- Member Type – Indicates whether user was a "registered" member (Annual Member, 30-Day Member or Day Key Member) or a "casual" rider (Single Trip, 24-Hour Pass, 3-Day Pass or 5-Day Pass)
Note that data is preprocessed to remove any trips that are taken by staff as they service and inspect the system, trips that are taken to/from any of our “test” stations at our warehouses and any trips lasting less than 60 seconds (potentially false starts or users trying to re-dock a bike to ensure it's secure).

Data Preprocessing: In order to better prepare the data, I took the followings data manipulations such as handling missing values, removing outliers, encoding categorical variables, and dropping unnecessary column such as 'Bike Number' to prepare the data for model training.

Feature Engineering: On top of the data preprocessing, I added extra columns into the dataframe that are: 'time_of_day', 'day_of_week', 'month', 'year' to capture the seasonality and trend of bike ride in terms of time of day, day of week, and monthly in 13years span. 

Data Split: Clarify how the data was split into training and testing sets. Specify the ratio or methodology used for partitioning the data to ensure unbiased evaluation and accurate performance assessment of the machine learning model.

Data Exploration and Visualization: Highlight any data exploration and visualization techniques employed to gain insights into the data distribution, correlations between variables, or trends over time. This could include interactive plots, charts, or statistical summaries to provide a comprehensive understanding of the data.

Data Limitations: In terms of the data limitation, the Capital Bikeshare service has a huge missing data from March 2020 until December 2020. This will affect the monthly basis analysis because the dataset is missing a year worth trip data from March to December in 2020, while January and February has data.  
In order to handle this issue, I took the method of both Time-based Subsetting and Ensemble Models. Time-based Subsetting excludes the period with missing data from the training and evaluation of the deep learning model. By focusing solely on the available data from previous years and omitting the COVID-affected period, the model can be trained and tested on a more consistent and reliable dataset.
On the other hand, Ensemble Models combine multiple models to make predictions. By training separate models on different subsets of the data (e.g., one model with pre-pandemic data only, another model with post-pandemic data), you can leverage the strengths of each model while mitigating the impact of missing data in a specific period.

----------------------------------------------------------------------------------------------------------------------
## Deep Learning Model:
For the prediction of the total trip count in the Capital Bike Share bike sharing service dataset, I considered using followings deep learning models with LSTM and took the folowings steps:
Here's an approach using LSTM for predicting the total trip count:

1. Preprocess the Data:
  - Convert categorical variables (such as 'Start station', 'End station', and 'Member type') into numerical representations using techniques like one-hot encoding or label encoding.
  - Split the dataset into training and test sets.

2. Build the LSTM Model:
  - Design an LSTM model architecture using frameworks like TensorFlow or Keras.
  - Define the input shape based on the number of features and the sequence length.
  - Add LSTM layers with an appropriate number of units.
  - Add other layers like Dense layers or dropout layers for regularization.
  - Compile the model with an appropriate loss function and optimizer.

3. Train the LSTM Model:
  - Fit the model on the training data.
  - Specify the batch size and the number of epochs for training.
  - Monitor the training progress and adjust the hyperparameters as needed.

4. Evaluate the LSTM Model:
  - Make predictions on the test set.
  - Calculate evaluation metrics such as mean squared error (MSE) or root mean squared error (RMSE) to assess the model's performance.

----------------------------------------------------------------------------------------------------------------------
## Choice of LSTM Model:
For the prediction of the total trip count in the Capital Bike Share bike sharing service dataset, LSTM (Long Short-Term Memory) networks are chosen because it is commonly used in sequence modeling task such as time series analysis, where there are long-term dependencies or patterns in the data. LSTM networks are a type of recurrent neural network (RNN) that can effectively capture and model such long-term dependencies.

In the context of predicting the total trip count in the Capital Bike Share bike sharing service dataset, LSTM was suggested as a deep learning model option because of the following reasons:

Sequential Nature of Data: The bike sharing service dataset exhibits temporal patterns and dependencies. The total trip count can vary based on factors such as the time of day, day of the week, and month. LSTM networks are designed to capture such sequential dependencies in the data, making them suitable for modeling time series or sequential data.

Memory Cell Architecture: LSTM networks incorporate memory cells that can retain information over long periods, allowing them to capture long-term dependencies in the data. This is particularly beneficial when predicting time series data, where past observations may have a significant influence on future outcomes.

----------------------------------------------------------------------------------------------------------------------
## Choice of Random Forest Regressor Model:
Random Forest Regressor is chosen as part of the Ensemble model with LSTM because:
Complementary strengths: LSTM is a powerful deep learning model that can capture sequential patterns in the data, while Random Forest Regressor is a versatile and robust ensemble model that can handle non-linear relationships and capture complex interactions between features.

Diversification: LSTM and Random Forest Regressor are fundamentally different models, so combining them in an ensemble allows for diversification and reduces the risk of relying on a single model's performance.

Robustness to outliers: Random Forest Regressor is less sensitive to outliers compared to LSTM, making it a valuable addition to the ensemble when dealing with noisy or outlier-prone data.

Interpretability: Random Forest Regressor provides feature importance measures, which can offer insights into the relative importance of different features in predicting the total trip count.

Improved predictive performance: Ensemble models have the potential to provide more accurate predictions by leveraging the strengths of multiple models, leading to improved overall performance and generalization ability.

----------------------------------------------------------------------------------------------------------------------
## Running the Ensemble Models:
```
# Import necessary libraries for running LSTM model and RF model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
```
```
# Assign independent and dependent variables based on 
X_pre_pandemic = pre_pandemic_data[['Start station number','End station number', 'Member type','day_of_week', 'month', 'year']]
y_pre_pandemic = pre_pandemic_data['Total trip count']
X_post_pandemic = post_pandemic_data[['Start station number','End station number','Member type','day_of_week', 'month', 'year']]
y_post_pandemic = post_pandemic_data['Total trip count']

# Define the column transformer for one-hot encoding
categorical_cols = ['Member type', 'day_of_week', 'month']
preprocessor = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_cols)],
    remainder='passthrough'
)

# Apply the column transformer to encode the categorical columns for pre-pandemic data
X_pre_pandemic_encoded = preprocessor.fit_transform(X_pre_pandemic)
# Convert the encoded data to a dense matrix
X_pre_pandemic_encoded = X_pre_pandemic_encoded.toarray()

# Scale the features for pre-pandemic data
scaler = MaxAbsScaler()
X_pre_pandemic_scaled = scaler.fit_transform(X_pre_pandemic_encoded)

# Define hyperparameters
lstm_params = [
    {'units': 50, 'epochs': 50, 'batch_size': 32},
    {'units': 100, 'epochs': 100, 'batch_size': 64},
    {'units': 200, 'epochs': 100, 'batch_size': 128}
]

rf_params = [
    {'n_estimators': 100},
    {'n_estimators': 200},
    {'n_estimators': 300}
]
```
```
# Perform model training and evaluation for each set of hyperparameters
results = []

for lstm_param in lstm_params:
    for rf_param in rf_params:
        # LSTM Model
        # Split the pre-pandemic data into train and test sets
        # Split the pre-pandemic data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_pre_pandemic_scaled, y_pre_pandemic, test_size=0.2, random_state=42)
        
        # Normalize the data
        scaler = MaxAbsScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reshape the input data for LSTM
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(lstm_param['units'], activation='relu', input_shape=(1, X_train_scaled.shape[1])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Train the LSTM model
        model.fit(X_train_reshaped, y_train, epochs=lstm_param['epochs'], batch_size=lstm_param['batch_size'], verbose=0)

        # Evaluate the LSTM model
        y_pred_lstm = model.predict(X_test_reshaped)
        mse_lstm = mean_squared_error(y_test, y_pred_lstm)

        # Ensemble Model
        # Train a Random Forest regressor on the post-pandemic data
        model_rf = RandomForestRegressor(**rf_param, random_state=42)
        model_rf.fit(X_post_pandemic, y_post_pandemic)

        # Predict using the Random Forest regressor
        y_pred_rf = model_rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)

        # Combine the predictions using simple averaging
        y_pred_ensemble = (y_pred_lstm + y_pred_rf) / 2
        mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)

        # Store the results
        result = {
            'LSTM Parameters': lstm_param,
            'Random Forest Parameters': rf_param,
            'LSTM MSE': mse_lstm,
            'Random Forest MSE': mse_rf,
            'Ensemble MSE': mse_ensemble
        }
        results.append(result)
```
