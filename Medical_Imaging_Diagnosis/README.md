# Chest X-Ray Image Diagnosis

The first approach to finding the diagnosis of patients looked at Convolutional Neural Networks. The corresponding code and notebook can be found [here](https://github.com/ndhers/My-Portfolio/blob/main/Medical_Imaging_Diagnosis/CNN-Part%201-1.ipynb).
An example of what the data looks like is included here (patient with cardiomegaly):

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main//blob/cardiomegaly.png)

Note that only images were used as input to predict illnesses and other metadata was disregarded. However, future work would look at incorporating information on the patients such as their age, gender or other for more accurate predictions.

The data was first imported and target variables were pre-processed. To enlarge the training data, the training data was augmented, while ensuring that the inherent meaning of data/targets is not compromised. Data was then into train, validation and test sets. Here, the tricky part was to avoid group data leakage where a single patient's X-rays could be found in different sets, therefore cheating and helping the algorithm make better predictions.
For this reason, it was key to keep track of patients id's.

EDA and data loading during training was streamlined through the use of tensorflow image data generator methods that increase training speed and leverage data caching. 

EDA showed class imbalance with illnesses being more present than others, as show in the figure below. 

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Medical_Imaging_Diagnosis/imgs/class_imb.png)

A weighted loss function was designed to account for this imbalance.

For modeling, I used a pre-trained DenseNet121 on ImageNet from Tensorflow, freezing the first 50 layers and only training the rest as well as an added logistic layer for our classification purpose. Using Early stopping, I only needed 6 epochs to train.
The resulting ROC curve indicating relatively good performance is shown below:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Medical_Imaging_Diagnosis/imgs/roc_curve.png)

Overall, this CNN model has decent performance, since the average AUC score is 0.687. I noticed that some diseases are easier to predict because of either more training examples for them or the diagnosis is more visually detectable.

Using Saliency maps, we were able to determine which pixels in the inputs are mostly responsible for driving the classification. 

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Medical_Imaging_Diagnosis/imgs/grad_cam.png)

Looking at the above, it would seem like grad_cam tends to look at the correct input image region to make a decision for cardiomegaly (enlarged heart). However, Smooth Grad was more looking at the lung area rather than the heart. Therefore, there is a concern that the model is learning shortcuts to make diagnosis rather than following a medically sound approach. 
One should consider working on more examples and tuning the model further (less frozen layers) for better generalization in the future.

Finally, we played around with a [visual transformer](https://github.com/ndhers/My-Portfolio/blob/main/Medical_Imaging_Diagnosis/VIT-Part%202-1.ipynb), using the pre-trained vitb16 from Tensorflow. The goal was to see if we could leverage attention in the context of image classification. The resulting ROC curve shown below shows decent but worse performance compared to the CNN above. The vision transformer is not able to capture as well contextual pixel information.

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Medical_Imaging_Diagnosis/imgs/roc_curve2.png)

In conclusion, the above models can be leveraged by doctors and nurses to make faster decision. However, they are limited in performance and saliency maps show that they do not always focus on the right pixels to make decisions. As such, they should only be used as tools and with care.
