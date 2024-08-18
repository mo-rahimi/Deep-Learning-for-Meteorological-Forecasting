# Deep-Learning-for-Meteorological-Forecasting
Deep Learning for Meteorological Forecasting: Advanced Weather Prediction Using LSTM Networks


     
## Table of Contents

<img src="Images/try3_pred.png" align="right"
     alt="Size " width="510" height="560">
     
1. [Overview](#overview)
2. [Data Collection](#data-collection)
   - [Data Sources](#data-sources)
   - [Error Resolution](#error-resolution)
3. [Data Cleaning for LSTM](#data-cleaning-for-lstm)
   - [Datetime Indexing](#datetime-indexing)
   - [Cyclical Feature Encoding](#cyclical-feature-encoding)
   - [Wind Vector Calculation](#wind-vector-calculation)
   - [Data Splitting](#data-splitting)
4. [Model Development](#model-development)
   - [Input and Output Arrays](#input-and-output-arrays)
   - [Standardization](#standardization)
   - [Loss Functions](#loss-functions)
5. [Results and Evaluation](#results-and-evaluation)
6. [Installation](#installation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)

## Overview
This project utilizes Long Short-Term Memory (LSTM) networks for weather prediction, specifically forecasting temperature, humidity, and UV index based on historical data from 2000 to 2023. All datasets are sourced from King's Park in Hong Kong to enhance accuracy.

Initially developed for learning purposes, this repository compiles various models and architectures I've explored, making it a valuable resource for others interested in deep learning.

LSTM networks are particularly effective for this task as they can capture complex patterns in time-series data. Historical weather data is organized into fixed-length sequences—specifically, `a sequence length of 7 days`. This allows the model to predict weather conditions for `the next 1, 2, or 4 days` based on the preceding week. The output sequences include the forecasted values for `temperature`, `humidity` and `UV index`.

## Data Collection
The dataset for this project is sourced from various resources on the `Hong Kong government data portal` and the `CSDI website`. 

**Key details include:**
- The Date column is created by merging the year, month, and day columns into a single datetime object, which acts as the primary key for dataset integration.
These datasets encompass a wide range of meteorological and spatial data, facilitating comprehensive weather analysis.

### Data Sources
1. Daily mean amount of cloud      [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-amount-of-cloud)
2. Daily total bright sunshine     [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-bright-sunshine )
3. Daily total evaporation         [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-evaporation)
4. Daily global solar radiation    [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-global-solar-radiation)
5. Daily mean relative humidity    [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-relative-humidity)
6. Daily mean pressure             [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-pressure )
7. Daily prevailing wind direction [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-prevailing-wind-direction)
8. Daily total rainfall            [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-rainfall)
9. Daily maximum mean UV index     [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-maximum-mean-uv-index)
10. Daily mean wind speed          [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-wind-speed)

### Error Resolution
Errors arose from incorrect date values in some datasets, particularly in the `df_mean_pressure DataFrame`. Entries outside the valid date range (January 1, 2000, to December 31, 2023) were removed.

When working with individual datasets, you may encounter several issues, one of which is the presence of incorrect date values. For example, some months contained more rows than expected, such as having 30 rows for a 29-day month. To address this, I identified the problematic DataFrame and corrected the discrepancies.

Given that there are 14 datasets with over 300,000 rows in total, I found it more efficient **to create the date column for each DataFrame individually**. After analysis, I confirmed that the df_mean_pressure DataFrame was problematic. 

## Data Preprocessing 

### Datetime Indexing
The `Date` column was converted to a datetime object and set as the index, while the original `Date` column was dropped.

### Cyclical Feature Encoding
New columns representing the cyclical nature of time (`Year sin` and `Year cos`) were created to help the model understand seasonal patterns, which is essencial for for time series prediction.

```pyton
df = pd.read_csv("Weather_HK_00.csv")
df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')   # Handel the column Date
df = df.iloc[:,1:]  # We do not need the column date any more
df['Seconds'] = df.index.map(pd.Timestamp.timestamp)

# Convert column date to 2 new columns 'Year sin' and 'Year cos'
day  = 24*60*60
year = (365.2425)*day
df['Year sin'] = np.sin(df['Seconds'] * (2 * np.pi / year))
df['Year cos'] = np.cos(df['Seconds'] * (2 * np.pi / year))
df = df.drop('Seconds', axis=1)  # We do not need column "Seconds" any more
```
<img src="Images/time_of_Year_signal.png" alt="Overview of the project" width="50%">

### Wind Vector Calculation
Wind direction was transformed into x and y components (`mean_wind_x` and `mean_wind_y`) to improve the input representation of wind data.

```python
# Calculate the max wind x and y components.
df['mean_wind_x'] = mean_wind_speed*np.cos(wind_direction_rad)
df['mean_wind_y'] = mean_wind_speed*np.sin(wind_direction_rad)
```

```python
plt.hist2d(df['mean_wind_x'], df['mean_wind_y'], bins=(50, 50), vmax=400)
plt.colorbar()
plt.xlabel('Wind X [km/h]')
plt.ylabel('Wind Y [km/h]')
ax = plt.gca()
ax.axis('tight')
plt.savefig('wind_x_y.png', format='png')
plt.show()
```
<img src="Images/wind_x_y.png" alt="Overview of the project" width="70%">

### Data Splitting
The dataset was divided into training (70%), validation (20%), and test (10%) sets for effective model training and evaluation.

## Model Development

### Input and Output Arrays
The df_to_X_y function converts the DataFrame into input (X) and output (y) arrays, employing a window size of 7 or 14 days to forecast the weather for the next 1, 2, or 4 days.

For example, in the code snippet below, we demonstrate how to predict the weather for the next 2 days:

```python
def df_to_X_y_days(df, window_size=7):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size - 1):  # Subtract 1 to account for the extra day in prediction
        row = [r for r in df_as_np[i:i + window_size]]
        X.append(row)
        label_day_1 = [df_as_np[i + window_size][2], df_as_np[i + window_size][6],
                       df_as_np[i + window_size][11]]
        label_day_2 = [df_as_np[i + window_size + 1][2], df_as_np[i + window_size + 1][6],
                       df_as_np[i + window_size + 1][11]]
        label = label_day_1 + label_day_2  # Combine the labels for both days
        y.append(label)
    return np.array(X), np.array(y)
X_days, y_days = df_to_X_y_days(df)
X_days_train, y_days_train = X_days[:6000], y_days[:6000]
X_days_val, y_days_val = X_days[6000:7800], y_days[6000:7800]
X_days_test, y_days_test = X_days[7800:], y_days[7800:]
```

### Standardization
Features were standardized using the mean and standard deviation derived from the training dataset, while excluding cyclical features and wind vector components.

`It's important to normalize the evaluation and test datasets using the mean and standard deviation from the training dataset to avoid common pitfalls.`

### Loss Functions
Mean Squared Error (MSE) was chosen as the loss function for training, while Mean Absolute Error (MAE) was used for evaluating model performance.

## Results and Evaluation
The model's performance was assessed using Mean Squared Error `(MSE)` and Mean Absolute Error `(MAE)` metrics, offering insights into its predictive accuracy across various weather parameters.

As illustrated in the graphs below, there is room for improvement. For instance, after epoch 25, the model begins to overfit, indicating that training should be halted at that point.

`Feel free to clone this repository and enhance the model further!`

<div style="display: flex; justify-content: space-around;">
    <img src="Images/rmse.png" alt="Image 1" width="46%">
    <img src="Images/rmse.log.png" alt="Image 2" width="46%">
   
</div>

## Installation
To set up this project, clone the repository and install the required Python packages:

```bash
git clone https://github.com/mo-rahimi/Deep-Learning-for-Meteorological-Forecasting.git

```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License
This project is licensed under the MIT License. 

## Contact
For inquiries, feedback or further information, please contact me🙂 at m.rahimi.hk@gmail.com

