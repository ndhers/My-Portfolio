# Image Colorization

The corresponding code and notebook can be found [here](https://github.com/ndhers/My-Portfolio/blob/main/Image_Colorization/code.ipynb). The notebook is organized in logical sequential manner.
For this project, the data comes from the Chinese University of Hong Kong, available [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). It contains about 200,000 color images of celebrity faces. Below is a sample from that dataset:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/example2.png)

TensorFlow Dataset functionalities were used to optimize data loading, augmentation and training processes. For each image in the dataset, we can convert the pixels from RGB color space to LAB colorspace, where the L channel (light intensity) can be used as input and the a,b channels as target variables.
An example is shown below:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/example3.png)

The training data was augmented to increase the variance of the training dataset and eventually improve model performance on unseen data. These were performed without compromising the semantic meaning of the inputs.

The performance of two different models was analyzed. The first one is built/trained from scratch. Due to limited computational resources, I kept the architecture simple and the number of trainable parameters small. Inspired by the [paper](https://dl.acm.org/doi/abs/10.1145/2897824.2925974),
It follows an auto-encoder structure to better learn the representation of the image. The autoencoder architecture is slightly modified for this task by customizing the output layer and adding skip connections. 
The output layer is set so that we obtain 2-channel images for both a and b channels. The output is then scaled to [-1,1] using tanh function to match the scale of our target variable. Finally, skip connections helped with gradient flow and with capturing the correct semantic meaning of the input image.
The figure below shows the described architecture. Note that dimensions are slightly off and readers are invited to consult the [notebook](https://github.com/ndhers/My-Portfolio/blob/main/Image_Colorization/code.ipynb) for more accurate dimensions. 

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/architecture.png)

The second model leverages pre-training of a ResNet50 model using ImageNet. By adjusting the input and output dimensions while freezing most of the upstream layers, the model was able perform on this task of colorization.

Both models performed equivalently on the validation sets, so only the custom model (non pre-trained) was tested on unseen data. Using many different test examples, the model performed reasonably well and captured generally well the main colors.
However, the colors predicted were not as vivid as the ground truth colors. This can be attributed to the relatively small size of both the model and the training data. Also, the scale of the loss output is relatively small and therefore encourages the modelto achieve small losses by predicting grey-tone colors.

Below are a couple examples, comparing ground truth to predicted colors.

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/cnn1_output.png)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/cnn1_output2.png)
