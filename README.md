# Unsupervised-Anomaly-Detection-of-Winter-Storm-Outages

Repository Structure
--------------------

    |- -1002.ipynb           # Singular zip code analysis
    |- Austin_hourly.csv, Austin_weather_daily.csv, buildingInfo.csv. # Data Sources
    |- requirements-.txt  # requirements
    |- GENERATING DATA (1).ipynb # ETL Script for generating data
    |- Graphs.py # Scripts for generating graphs
    |- LOF.ipynb  # Local Outlier Factor Model
    |- agg_and_heatmap_functions.py # Data viz's functions and heatmap functions
    |- check_inf_servicepoint.ipynb         # Checking service points with inf values of % change between measured and predicted energy
    |- commercial.ipynb      # Modeling for commercial buildings
    |- covid-praveen_Baseline.ipynb   # Baseline model for covid-19 electricity market planning
    |- covid-praveen_final_model.ipynb # Optimized model for covid-19 electricity market planning
    |- fig_gen.py            # Script for figures generation
    |- full_servicepoint.ipynb          # Analysis and modeling for all zip codes
    |- README.md        # the top-level description of content and commands to reproduce results, data download instructions
    |- model_fns.py   # Script for all the model, feature engineering, and preprocessing functions.
    |- residential.ipynb#   # Modeling for Residential buildings
    |- winter_storm_predictions-single_zipcode_iqr_zipcode_selection.ipynb- zipcode selection based on good or bad zip codes( for publications)


## Project Goal

During the Winter Storm Uri disaster in TX in 2020, there were a lot of outages across zip codes. The main reason for this was, that the state was underprepared and didn't satisfy the supply and demand. So, a lot of zip codes went blacked out and experienced outages during the event. So to avoid these in the future, we prepared an unsupervised anomaly detection model using ensemble learning that will identify the vulnerable zipcodes during winter storms and also tell us the magnitude of energy that should be provided to that zipcode to avoid outage. So ultimately we can use these results and prepare the austin city zipcodes for future winter storm disasters.

## Data Sources

We got the data from austin energy. The energy was in the form of a univariate time series with energy measured at 15 15-minute scales. The period is from 2017 to 2021 for the city of Austin. The zip code has other metadata such as smart meters, zip code area, etc.

## Data Pre-Processing and Feature Engineering

After getting data we performed data cleaning. We used both domain knowledge, IQR, and visualizations to remove bad energy values. We also aggregated the 15-minute scale to larger scales like 30 mins, 1 hour, 1 day, etc for much further analysis. We applied normalization and then created time features so our regression and random forest models could perform better. We also added self-attention to our autoencoder so that the model gives different weights to different parts of the time series in a day and for K-means clustering we used the elbow method to determine the number of clusters.

## Models Used

our goal is to identify the zipcodes that faced outages which are typically anomalies, because the energy profile during the event will be completely different from normal cases during the case of outage. So to identify these anomalies, we used an ensemble learning of k means clustering, local outlier factor, and lstm-based autoencoder. Majority voting was used. The reason for using these models was because since we dont know true outages, it's a form of unsupervised learning tasks and these models can perform without true labels. Hence these models were chosen. The models are diverse from each other to reduce variance and overfitting.  To find the anomalies that are true outages, this anomaly detection pipeline was used and zipcode is considered an outage if the majority of models mark it as an outage, and to identify the magnitude of the percentage difference of measured energy from predicted energy, we used models like regression and random forest. This model tells us how much the measured energy lacks from the predicted energy during the event. So by getting results from anomalies and the magnitude of energy difference, we can make informed decisions on which zip codes have the most possible chance of outage so we can prepare for the future

## Model Evaluation

So even though it's an unsupervised task, to have an additional validation of our results and models, we used domain knowledge and expertise to set some constraints that will classify if the zipcode has experienced an outage, like pseudo true labels. So after getting pseudo true labels, the metric we used in recall score. The reason for using the recall score is we want to focus on false negatives because we dont want to classify zip codes that did not experience outage, but it experience outage. Also, we used k means clustering as a baseline method to measure how much our ensemble model learning improved and the ensemble learning improved the score by 35%

## Visualizations of findings.(ANALYTICAL DASHBOARD)

![image](https://github.com/prav0761/Unsupervised-Anomaly-Detection-of-Winter-Storm-Outages/assets/93844635/0b14416f-3be7-4aee-9a66-b04e46f2165c)

![image](https://github.com/prav0761/Unsupervised-Anomaly-Detection-of-Winter-Storm-Outages/assets/93844635/d8483e64-5c12-4450-8488-e4bfe5ae9864)

Discussions
The analysis of Figure 4 and Figure 5 indicates that the period from Feb 11th to Feb 14th did not show any significant
signs of possible outages, based on both our classification
and regression models. However, the dates from Feb 15th
to Feb 17th showed a strong indication of possible outages,
with a large number of outliers detected by the classification model and a higher magnitude in load lost percentage by the regression model, with Feb 16th showing the
highest magnitude. The dates of Feb 18th and Feb 19th
showed recovering signs from outages, returning to normalcy. Our results are reliable as they coincide with the
timeframe of the actual power crisis in Texas, which lasted
from Feb 15th to Feb 18th (Adam Barth, 2021) where Feb
15th, Feb 16th, Feb 17th dates showing an increase in load
lost and our results also signaling the same. Our results
also show that around 60% of residential meters exhibit
possible outages on Feb 16th by calculating the percentage
of meters corresponding to residential zipcodes with possible outages to total meters in residential zipcodes. This result also agrees with (City, 2021) report where they discussed that more than 40% of customers lost power on
Feb 16th. Additionally, our classification and regression
models provided consistent signals, with an increase in
outliers indicating a higher load lost percentage, thus predicting strong signs of possible outages. Our study also
provided insights into the magnitude of the load lost percentage for each zipcode on each day, which can aid in
supply-demand planning for future extreme weather scenarios. Furthermore, our study developed a framework to
identify the zipcodes more affected by the outages, which
can help allocate resources and prioritize relief efforts. To
address the uncertainty and validate our results due to the
absence of ground-truth labels, we developed an ensemble approach for our classification model and computed
95% confidence intervals for our linear regression model
to calculate the lost load percentage. We also included a
random forest model to compare the performance and increase the reliability of our linear regression model. These
methods increase the reliability of our results and address
the limitation of model validation in unsupervised studies


## Results
Based on above figures,Our study sheds light on the impact of Winter Storm Uri
on residential zipcodes in Austin and provides valuable insights into demand losses and identifying vulnerable zipcodes in Austin. Our unsupervised ensemble classification and regression models proved effective in detecting
possible outages and calculating load lost, respectively.
Our findings indicate that the period from Feb 15tth to Feb
17th was the most critical time for possible outages, with
a higher magnitude of the load lost percentage detected on
Feb 16th. This information can be used to aid in supplydemand planning for future extreme weather scenarios
and prioritize resource allocation for vulnerable zip codes.
Overall, our study highlights the importance of proactive
measures and effective strategies to ensure the reliability
of the electricity grid during extreme weather events. The
potential application of our study is our methodology can
serve as a reference for future studies in building energy
domains with unsupervised anomaly detection tasks with
no ground truth and also can serve as a reference to studies
with fewer data. Our results and data analysis can inform
policymakers, utility companies, and other stakeholders in
developing effective strategies to minimize the impact of
future extreme weather events on residential buildings.



