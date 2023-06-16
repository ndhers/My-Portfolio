# NLP Approach to Time Series Analysis

In this project, the data comes from PhysioNet and provides information on patients in ICU, and whether they survived or not. The goal of this project is to predict their chance of survival.
Often, such medical data is in a sparse format containing many missing values and is irregularly sampled. 
For example, a patient’s vital signs may be measured at irregular intervals depending on their condition and the availability of medical staff. 
In such cases, using regular, evenly spaced time intervals for analysis may not accurately capture the underlying patterns and trends in the data.
Working with irregularly sampled time series data can be complicated when it comes to using machine learning. 
This is because most machine learning algorithms are designed to work with regularly spaced data, and may not be able to effectively handle data that is irregularly spaced.

In this project, multiple pre-processing and modeling combinations were studied in order to accurately model multivariate, irregularly sampled and sparse time-series data.

The code can be found in this [notebook](https://github.com/ndhers/My-Portfolio/blob/main/NLP_TimeSeries/model.ipynb).

For pre-processing, two different approaches were analyzed: index-based masking and uniform padding:
<ul>
  <li>For index-based masking, only the latest fifty timestamps of a given patient are selected and the corresponding timestamps are used as indices. The intuition behind selecting the latest timestamps is that the later measurements contain more information and are better indicators of a patient’s death.
    Variables and timestamps that have missing values are masked by the value ”-1”. Thereafter in the model, a masking layer is used to inform the model that an observation is missing.
  <li>Uniform padding consists in taking the maximum length of all irregularly-sampled time series across all patients. Back padding is then applied on all sequences to match this maximum length with zeros. In the modeling part, padding tokens are set as "0" to keep track of padded values.</li>
</ul>

Because the data is imbalanced with more patients that survived, it was important to put extra emphasis on those patients that were incorrectly predicted to survive when they in reality did not. To that end, different loss functions and classification thresholds were tested. 
In the end, the threshold was lowered from the default 0.5 to 0.3 to provide more conservative outputs. Note that probabilities above 0.3 would mean that patients are predicted to not survive.

For the modeling part, the performance of two different appraoches was compared: a bi-directional LSTM classifier and a Transformer Encoder classifier. The reasoning here is that we need to learn information from the sequential nature of the data. Because time is a factor and the evolution of variables has a meaningful impact on a patient’s chances of survival, borrowing from Natural Language Processing and using recurrent models makes sense.

Because of self-attention, the Transformer encoder approach also makes sense and has the advantage of allowing for parallel training, which allowed for longer training and more tuning. The output layer of the encoder was modified to output a singe probability number.

Both models were tried with both pre-processing techniques and the results are shown in the table below:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/NLP_TimeSeries/imgs/results.png)

From the above, we can see that the best-performing approach is the one involving padding and the transformer encoder. Padding had the advantage of retaining more information than the masking approach, and the transformer was able to attend to any part of that information that it deemed important for prediction.

