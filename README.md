# Data-Mining
Data Mining Final Project

Objective:

To predict the likelihood of a client subscribing to a term deposit based on their demographics and other attributes, of a Portuguese retail bank using its data from 2008-2013. 

Overview:

The Business problems we found using dataset is that 1)Bank's current marketing campaigns have suboptimal resource allocation and low conversion rates` 2)Need to improve efficiency of campaigns by targeting right clients with personalized offers And our goal was to increase bank’s subscription rate for term deposits.

We have built a Predictive model to estimate client subscription likelihood. We then identified key factors influencing subscription decisions and built a propensity score model to identify high-value leads. We have optimized marketing campaigns based on model insights and created segment-specific models for tailored campaigns.

Data Description

a) The dataset contains real data collected from a Portuguese retail bank, from May 2008 to June 2013, in total of 52944 phone contacts.

b) Only 5289(11.7%) records subscribed to the term deposit

c) In the modeling phase, the dataset is reduced from the original set(150 features) to 17 relevant features that are used in ML models.

d) Input variables are age, job, marital status, education, and more.

e) Target variable is whether a client subscribed to term deposit or not

Results

Gradient Boosting Metrics Accuracy: 0.91 Precision: 0.68 Recall: 0.40 F1-score: 0.51 ROC AUC: 0.93

Insights

Most Important features

a) Identified the most important features such as the duration of the last call, bank balance, day and more b) Best month to contact: May c) Best month day to contact: 20th

Accuracy for Segment Specific Models Based on Call Duration

a) Accuracy for short duration (less than or equal to 100 secs): 0.98 b) Accuracy for medium duration (greater than 100 seconds and less than or equal to 300 seconds): 0.90 c) Accuracy for longer duration (greater than 300 seconds): 0.77

Business Recommendations

Prioritize High-Value Leads Focus marketing efforts on the 1,882 high-value leads as they have the highest conversion potential. Develop targeted campaigns and offers for this segment.�
Optimize Contact Strategy Contact high-value leads in May, 20th of a month for optimal response rates.
Personalize Messaging Craft messaging focused on economic conditions based on importance of employment and interest rates.
Update Targeting Model Continuously update model with latest data on key factors like duration and bank balance.�
Track Performance Measure marketing KPIs for high-value segment to quantify impact. Monitor model performance to check for concept drift.
