# Ozone Level Detection Using Machine Learning Pipelines

This project aims to analyze and predict ozone levels to improve air quality management. Using historical data from the Houston, Galveston, and Brazoria areas, we will develop a model to detect peak ozone levels. The model will help in timely intervention and reducing health risks associated with high ozone levels.

The dataset can be accessed via this link - https://archive.ics.uci.edu/dataset/172/ozone+level+detection

# Publisher of the Dataset:
The UCI Machine Learning Repository is a collection of datasets widely used in the machine learning community. Hosted by the University of California, Irvine, it offers datasets from diverse domains, such as biology, medicine, and social sciences. These datasets are curated and documented, providing researchers with valuable resources for benchmarking algorithms and developing new techniques in machine learning and data mining.

# Dataset Introduction:
The dataset includes records from 1998 to 2004 for eight-hour and one-hour peak ozone levels. It contains 72 features and 2536 instances, focusing on atmospheric conditions like temperature, wind speed, and humidity.

# Coverage
Those data were collected from 1998 to 2004 at the Houston, Galveston and Brazoria area.

# Data Rights and Privacy:
The dataset is publicly available for research purposes with no specific data privacy concerns, but general data protection principles should be followed.

# Python Access
To access this dataset using python, we first need to install the ucimlrepo library using the command:
pip install ucimlrepo

Then, we run the following code snippet to import the dataset to into our code:

from ucimlrepo import fetch_ucirepo 
# fetch dataset 
ozone_level_detection = fetch_ucirepo(id=172) 
# data (as pandas dataframes) 
X = ozone_level_detection.data.features 
y = ozone_level_detection.data.targets 
# metadata 
print(ozone_level_detection.metadata) 
# variable information 
print(ozone_level_detection.variables) 

# Tables and Columns
we have two different subsets of data where one of them records 8 hours of ozone activity whereas the other records ozone activity for 1 hour.

 Below is a detailed explanation of each column in the dataset:

WSR0 to WSR7 (Wind Speed Readings):
Type: Continuous
* Description: These columns represent wind speed readings taken at various times throughout the day. Each column corresponds to a specific time period.

WSR_PK (Peak Wind Speed):
Type: Continuous
* Description: The peak wind speed during the day, calculated as the resultant or average of the wind vector.

WSR_AV (Average Wind Speed):
Type: Continuous
* Description: The average wind speed throughout the day.

T_PK (Peak Temperature):
Type: Continuous
* Description: The peak temperature recorded during the day.

T_AV (Average Temperature):
Type: Continuous
* Description: The average temperature recorded throughout the day.

T85, RH85, U85, V85, HT85 (850 hPa Level Readings):
Type: Continuous
* Description: These columns represent various atmospheric readings at the 850 hPa level, roughly 1500 meters above sea level. They include temperature (T85), relative humidity (RH85), east-west wind component (U85), north-south wind component (V85), and geopotential height (HT85).

T70, RH70, U70, V70, HT70 (700 hPa Level Readings):
Type: Continuous
* Description: Similar to the 850 hPa readings but taken at the 700 hPa level, roughly 3100 meters above sea level.

T50, RH50, U50, V50, HT50 (500 hPa Level Readings):
Type: Continuous
* Description: These readings are taken at the 500 hPa level, roughly 5500 meters above sea level.

KI (K-Index):
Type: Continuous
* Description: A measure used in meteorology to estimate the likelihood of thunderstorms.

TT (Total Totals Index):
Type: Continuous
* Description: An index used to predict severe weather conditions, including thunderstorms.

SLP (Sea Level Pressure):
Type: Continuous
* Description: The atmospheric pressure at sea level.

SLP_ (Change in Sea Level Pressure):
Type: Continuous
* Description: The change in sea level pressure from the previous day.

Precp (Precipitation):
Type: Continuous
* Description: The amount of precipitation measured.





