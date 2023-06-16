# Airfoil Performance Prediction 

This project can be subdivided into two projects: one involves a novel approach where a dual signal is propagated through a standard feedforward deep neural network; another involves leveraging computer vision and CNN's for deriving insights from airfoil image data.
Both projects used data from the University of Illinois at Urbana-Champaign that can be found [here](https://m-selig.ae.illinois.edu/ads/coord_database.html). This database contains a substantial amount of data where each row pertains to the geometry of a given airfoil (i.e. the 2D points coordinates).
Reconstructing one row and using cubic Spline interpolation, I was able to obtain a decent amount of data points per airfoil. I used the Xfoil software to label my dataset. The latter takes in as input airfoil geometries as well as different angles of attack and flow conditions and outputs drag and lift coefficients.
The image below is a sample taken from this database:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/airfoil.png)

## Dual Signal Approach

The corresponding code can be found in this [subdirectory](https://github.com/ndhers/My-Portfolio/blob/main/Aircraft_Performance/dual_cnn/). 

This neural network tries to predict both the aerodynamic coefficients and their partial derivative with respect to the airfoil geometry.
This added level of information improves performance and fastens training. To do so, we propagate a dual signal through all layers of the neural network, therefore 'enhancing' the level of information it carries forward for inferance and backward for training. This neural network is said to be "gradient enhanced". 

For more details, please consult this [poster](https://github.com/ndhers/My-Portfolio/blob/main/Aircraft_Performance/dual_cnn/DNicolas.pdf) I made that contains more information and visualization on this. 

## Computer Vision Approach

The corresponding code can be found in this [subdirectory](https://github.com/ndhers/My-Portfolio/blob/main/Aircraft_Performance/artificial_img/).

This approach is very different to the one above. Here, I created 'artificial images' (using this [script](https://github.com/ndhers/My-Portfolio/blob/main/Aircraft_Performance/artificial_img/CNN_Image_builder.py)) by plotting airfoils using cubic spline interpolation from the discrete 2D points. 
From there, I tweaked the images so they could incorporate the same input information that the previous approach had, namely the angle of attack and the freestream velocity. I rotated the airfoils based on angle of attack and I adjusted pixel intensity proportionally to the freestream velocity.

Using these images as input as well as the Xfoil generated aerodynamic coefficients as output, I was able to achieve better performance than the first approach. Most importantly, my training was much faster as the number of trainable parameters was much less since CNNs share weights through sliding filters.

