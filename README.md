# Deep-Learning-for-Meteorological-Forecasting
Deep Learning for Meteorological Forecasting: Advanced Weather Prediction Using LSTM Networks

## Table of Contents
1. [Overview](#overview)
2. [Data Collection](#data-collection)
   - [Data Sources](#data-sources)
   - [Date Handling](#date-handling)
   - [Error Resolution](#error-resolution)
   - [Nan Value Imputation](#nan-value-imputation)
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
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [License](#license)
10. [Acknowledgements](#acknowledgements)

## Overview
This project implements a weather prediction model using Long Short-Term Memory (LSTM) networks. The model forecasts weather conditions such as temperature, humidity, and UV index based on historical data from the King's Park weather dataset in Hong Kong, spanning from 2000 to 2023.

## Data Collection

### Data Sources
The data for this project is collected from various datasets available on the Hong Kong government data portal. The datasets include:
1. Daily mean amount of cloud      [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-amount-of-cloud)
2. Daily total bright sunshine     [Daily total bright sunshine](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-bright-sunshine )
3. Daily total evaporation         [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-evaporation)
4. Daily global solar radiation    [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-global-solar-radiation)
5. Daily mean relative humidity    [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-relative-humidity)
6. Daily mean pressure             [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-pressure )
7. Daily prevailing wind direction [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-prevailing-wind-direction)
8. Daily total rainfall            [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-total-rainfall)
9. Daily maximum mean UV index     [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-maximum-mean-uv-index)
10. Daily mean wind speed          [Link to Hong Kong Government portal](https://data.gov.hk/en-data/dataset/hk-hko-rss-daily-mean-wind-speed)



### Date Handling
The `Date` column was created by combining the `year`, `month`, and `day` columns into a single datetime object, which serves as the primary key for merging datasets.

### Error Resolution
Errors were encountered due to incorrect date values in some datasets. The problematic `df_mean_pressure` DataFrame was identified, and entries outside the valid date range (2000-01-01 to 2023-12-31) were removed.

### Nan Value Imputation
Missing values were addressed by calculating the mean of adjacent values and filling in gaps, resulting in a clean dataset for analysis.

## Data Cleaning for LSTM

### Datetime Indexing
The `Date` column was converted to a datetime object and set as the index, while the original `Date` column was dropped.

### Cyclical Feature Encoding
New columns representing the cyclical nature of time (`Year sin` and `Year cos`) were created to help the model understand seasonal patterns.

### Wind Vector Calculation
Wind direction was transformed into x and y components (`mean_wind_x` and `mean_wind_y`) to improve the input representation of wind data.

### Data Splitting
The dataset was divided into training (70%), validation (20%), and test (10%) sets for effective model training and evaluation.

## Model Development

### Input and Output Arrays
The `df_to_X_y` function was utilized to convert the DataFrame into input (X) and output (y) arrays, using a window size of 7 or 14 days to predict the next 1, 2, or 4 days of weather.

### Standardization
Features were standardized using the mean and standard deviation calculated from the training dataset, excluding cyclical features and wind vector components.

### Loss Functions
Mean Squared Error (MSE) was chosen as the loss function for training, while Mean Absolute Error (MAE) was used for evaluating model performance.

## Results and Evaluation
The model was evaluated using MSE and MAE metrics, providing insights into its predictive accuracy and performance across different weather parameters.

## Installation
To set up this project, clone the repository and install the required Python packages:

```bash
git clone https://github.com/yourusername/weather-forecasting-lstm.git
cd weather-forecasting-lstm
pip install -r requirements.txt
```

## Usage
To run the weather forecasting model, execute the following command:

```bash
python main.py
```

Make sure to adjust any configuration settings in the `config.py` file as necessary.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements
- Special thanks to the Hong Kong government for providing the weather datasets.
- Thanks to the open-source community for their contributions to machine learning libraries used in this project.



### Notes:
- Replace `yourusername` in the installation section with your actual GitHub username.
- Ensure that the `requirements.txt` file is included in your repository with all necessary dependencies.
- Customize any sections as needed to better fit your project's specifics.
