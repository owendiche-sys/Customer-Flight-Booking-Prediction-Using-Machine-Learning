**Author: Owen Nda Diche**



**Customer Booking Prediction Using Machine Learning**

**Project Overview**



This project applies machine learning techniques to predict whether a customer will complete a flight booking based on behavioural and travel-related features. The aim is to identify key factors that influence booking decisions and demonstrate how predictive analytics can support business and marketing strategies in the airline industry.



**Objective**



* Build a predictive model to classify booking outcomes
* Evaluate model performance using standard classification metrics
* Identify the most influential features affecting booking behaviour
* Translate model results into actionable business insights



**Dataset Description**



The dataset contains customer-level booking information, including:



* Number of passengers
* Sales channel
* Trip type
* Purchase lead time
* Length of stay
* Flight time and route
* Customer preferences (extra baggage, seat selection, meals)
* Target variable: booking\_complete (1 = booking completed, 0 = not completed)



**Tools \& Technologies**



* Python
* Pandas
* Scikit-learn
* Matplotlib / Seaborn
* Jupyter Notebook



**Data Preparation**



The following preprocessing steps were applied:



* Removal of irrelevant columns
* Handling missing values
* Encoding of categorical variables using one-hot encoding
* Splitting the dataset into training and testing sets



These steps ensured the dataset was suitable for machine learning modelling.



**Model Used**



A Random Forest Classifier was selected due to its:



* Strong performance on structured data
* Ability to model non-linear relationships
* Built-in feature importance for interpretation
* Robustness to overfitting



**Model Performance**



The model was evaluated using accuracy and classification metrics.



Results:



* Accuracy: 85.5%
* The model demonstrates strong predictive capability in distinguishing between booking and non-booking customers.



**Feature Importance**



The most influential features identified by the model include:



* Purchase lead time
* Previous booking behaviour
* Travel-related attributes (route, duration)
* Customer preference indicators (baggage, meals, seat selection)



These findings highlight the importance of customer behaviour in predicting booking outcomes.



**Business Insights:**



* Customers who plan trips earlier are more likely to complete bookings
* Behavioural and engagement variables are strong indicators of purchase intent
* The model can support targeted marketing and personalised offers
* Predictive insights can help improve conversion rates and customer engagement



**Conclusion**

This project demonstrates how machine learning can be used to analyse customer behaviour and support data-driven decision-making. The model achieved strong performance and provided meaningful insights that could be applied in real-world airline marketing and revenue optimisation scenarios.





