# Leveraging Computer Vision for Reading Barcodes

In this project, I wanted to build a workflow that would facilitate product lookup and save people time and resources. Instead of having to understand the intricacies of SQL and databases, one could upload an image
of the product they look to get information on and the model would be able to find the associated bar code and pull up the relevant data.

The workflow can be followed throughout this [notebook](https://github.com/ndhers/My-Portfolio/blob/main/Bar_Code/bar_code_nb.ipynb).

To do so, I used the famous segmentation algorithm YOLOv5, free of access on [Github](https://github.com/ultralytics/yolov5). 

I trained it on a labeled dataset of products and their respective bar code segmentations that followed YOLO target output conventions. The data is also freely available [here](https://www.kaggle.com/datasets/whoosis/barcode-detection-annotated-dataset).

We can then use the saved weights for inference for any custom input image. I used the following picture of my allergy meds:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/test_original.jpg)

After performing segmentation, here is the output:

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/test.jpg)

I then used a Python [package Pyzbar](https://pypi.org/project/pyzbar/) for reading bar codes. By extracting the sequence of characters and numbers, one can then easily look up the corresponding product information.

