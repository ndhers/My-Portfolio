# Airline Passenger Satisfaction

Please find the full notebook [here](https://github.com/ndhers/My-Portfolio/blob/main/Passenger_Satisfaction/Code.ipynb).

Data is available for free [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).

The code follows a standard data science workflow. After importing and cleaning the data, EDA is performed to better understand the statistics and distributions of the data. Feature importance and selection is accomplished and followed by modeling. The screenshots below highligh some of the key feature/target relationships. First with quantitative predictors:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Passenger_Satisfaction/imgs/targ_dist_quant.png)

Then, with categorical predictors:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Passenger_Satisfaction/imgs/targ_dist_cat.png)

In modeling, the performance of three models is compared: SVM, tree-based models and a simple feedforward neural network to the baseline Lasso logistic regression. Metrics of accuracy, sensitivity and specificity are used for comparison purposes. Confusion matrices, and evaluation metrics values are detailed in the full [notebook](https://github.com/ndhers/Portfolio/blob/main/Passenger_Satisfaction/Code.ipynb).

In conclusion, the bagging model seemed to have performed best on unseen data. From feature analysis, predictors like the type of travel (personal or business), the customer type (loyal or not) and the cabin class seem to have the largest impact on the dependent variable.
Surprisingly, features like 'Flight Distance' and 'Food and drink' are not among the most important predictors. From this, airlines could for example focus on targeting customer types with less satisfaction, while adopting different approaches for leisure or business customers.

