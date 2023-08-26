
# DS/DE Porfolio

Hi and welcome to my portfolio page. I try to maintain a list of some of the interesting projects I have worked on in the past. They span various Data Science areas and reflect my interests in the field. I love playing with all sorts of data and deriving insights from it. Most importantly, I like to learn and find out about new technologies and algorithms.

My background is in Mechanical Engineering (Bachelor) as well as Computer Science (Master). I like to take inspiration from these fields to work on interesting projects.

I can be reached at <nicolasdhers@g.harvard.edu> or through [LinkedIn](https://www.linkedin.com/in/nicolas-dhers/).

# Table of Contents
1. [Data Analysis](#data-analysis)
2. [Python Package for Automatic Differentiation](#python-package-for-automatic-differentiation)
3. [Machine Learning](#machine-learning)
4. [Deep Learning: Computer Vision](#deep-learning-computer-vision)
5. [Deep Learning: NLP](#deep-learning-nlp)

## Data Analysis 

### [Due Diligence Analysis for Fast-Food Chain Investment](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence/)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main//blob/due_dil.png)

The corresponding code and documementation can be found [here](https://github.com/ndhers/My-Portfolio/tree/main/Due_Diligence). In this project, I provide a complete analysis on eight american fast-food restaurant chains. By looking at several metrics and trends over the past couple years, I demonstrate which fast-food chain looks financially healthiest and provide an investment recommendation. 

The code, report and data can be found [here](https://github.com/ndhers/My-Portfolio/blob/main/Due_Diligence).

## [Python Package for Automatic Differentiation](https://github.com/ndhers/My-Portfolio/blob/main/Automatic_Differentiation/)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main//blob/full_graph.png)

The corresponding code and documementation can be found [here](https://github.com/ndhers/My-Portfolio/tree/main/Automatic_Differentiation). The main purpose behind this project is to showcase underlying mechanisms of automatic differention, a core concept used extensively in deep learning when training neural networks through backpropagation. Implementation follows standards of software engineering for package development.

`autodiff` is a modern, clean, fast implementation of automatic differentiation. More complete documentation can be found [here](https://github.com/ndhers/My-Portfolio/blob/main/Automatic_Differentiation/docs/documentation.md).

## Machine Learning

### [Predicting popularity of spotify playlist](https://github.com/ndhers/My-Portfolio/tree/main/Spotify_Playlist)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/spotify.png)

How can you get more people to follow your Spotify playlists? I like to create playlists for different moods and love sharing them with my friends. I decided to work on this project because of personal interests but also because it has a tangible business impact. 

The main goal is to predict the popularity of a Spotify playlist, using the number of followers as a measure of popularity. It was also interesting to find the inherent features of these playlists that make them popular. Spotify may be interested in what specific attributes yield a higher number of followers. This would in turn help attract users to the music platform while keeping current users satisfied.

Code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/Spotify_Playlist).

### [Predicting Airline Passenger Satisfaction](https://github.com/ndhers/My-Portfolio/tree/main/Passenger_Satisfaction)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main//blob/satisfaction.jpeg)

Now that the COVID-19 pandemic is (somewhat) subsiding, travel becomes an option again for many people around the world. Airline flights are a key component of any trip: having a poor flight experience means starting the trip in bad conditions. By expressing, their dissatisfaction, customers can impact future use of a given airline. 

In this project, the goal was to look at how flight and customer information can impact passenger satisfaction. By being able to predict passenger satisfaction, airlines can make informed decisions that can drive business.

The code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/Passenger_Satisfaction).

## Deep Learning: Computer Vision

### [Computer Vision in Healthcare](https://github.com/ndhers/My-Portfolio/tree/main/Medical_Imaging_Diagnosis)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main//blob/cardiomegaly.png)

Computer vision has made tremendous leaps in the recent years due to the development of more complex algorithms and the increase in computational power. Its use in healthcare has also grown and it has become an important tool in helping doctors make diagnosis using image data. I have always been passionate about computer vision and wanted to use this project as an opportunity to demonstrate how powerful it can be in such an important field. 

In this particular project, the goal is to determine which patients are sick and which are not based on X-ray images of their chest. If patients are sick, the output should state which pathology they suffer from (from a list of 15 different pathologies). Note that patients can suffer from multiple illnesses at the same time. By also highlighting which part of the X-rays drove the output, this model can help doctors make faster and more informed decisions. I would like to think that some of the medical tasks are repetitive and should be facilitated by AI with the end goal being to treat more patients in a better way.

The code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/Medical_Imaging_Diagnosis).

### [Computer Vision for Image Colorization](https://github.com/ndhers/My-Portfolio/tree/main/Image_Colorization)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/Image_Colorization/imgs/example.png)

In this project, the goal is to develop a tool able to recreate colorized images from black and white inputs. This can be particularly useful for anyone trying to better understand history through more accurate (and colorized) scene representations, therefore providing a better depiction of life before the modern area. In many instances also, we can find ourselves with only access to black and white visual data, in which case this model would come in handy as well. 

In this project, a novel approach is used and consists in downsampling the image to capture its main components before upscaling it again with skip connections from the input to better understand the input image content. This approach involves a custom convolutional network and takes inspiration from the ['Let there be Color!: Joint End-to-end learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification'](https://dl.acm.org/doi/abs/10.1145/2897824.2925974). This approach involves use of CIELAB color space. Another model serves as benchmark for comparison and is a modified pre-trained CNN. 

The code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/Image_Colorization).


### [Computer Vision for Airfoil Performance](https://github.com/ndhers/My-Portfolio/tree/main/Airfoil_Performance)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/airfoil.png)

This [directory](https://github.com/ndhers/My-Portfolio/tree/main/Airfoil_Performance) contains some research work I did on ways to optimize aircraft (more specifically airfoil) performance by predicting drag and lift coefficients. The airfoil is the tip of the wings and its shape is shown on the picture above.

To this day it is extremely expensive and close to impossible to model exactly the turbulent flow around aircrafts. Solving exactly the Navier-Stokes equations is too complex and requires making simplifying assumptions. High-fidelity approaches have typically been demonstrated through gradient-based optimization techniques, but these approaches also require extensive computational resources.

The goal of this research was to explore deep learning alternatives to finding aerodynamic coefficients of airfoils. These would allow for faster estimates of these parameters, which are key in characterizing aircraft performance.

The code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/Airfoil_Performance).


### Computer Vision For Reading Barcodes

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/blob/barcode.png)

I was asked at work to find a fast and easy way to filter a database of products for people with no data science or engineering background. The target audience would therefore be people not familiar with SQL syntax and querying. The project here is obviously different from what was implemented at work but shows some of the tools that can be used for this particular application.

Since all products' bar code information can easily be stored in a database, I focusing on figuring out a new way of searching for products. This would involve having people simply submit a picture of the products they wish to get information for and the application would automatically segment the input, read the bar code and pull up the relevant data. 

More details can be found [here](https://github.com/ndhers/My-Portfolio/tree/main/Bar_Code).


## Deep Learning: NLP

### [NLP For Irregularly Sampled Time Series](https://github.com/ndhers/My-Portfolio/tree/main/NLP_TimeSeries/)

![img not available](https://raw.githubusercontent.com/ndhers/My-Portfolio/main/NLP_TimeSeries/imgs/transformer.png)

For numerous academic projects on time-series analysis, we had to deal with data that was irregularly sampled. This is especially true in the field of healthcase or when dealing with survey data. This poses a large problem for time series analysis and forecasting, as deep learning models often use fixed-dimension representations. In this project, the dataset consists in time series data from patients in ICU and the goal is to predict whether they would survive or not. The collected data per patient spans multiple days and multiple variables (e.g. temperature, weight, BP, etc). Note that patients, because they stay in ICU for various amounts of time, have different amounts of data available. 

Dealing with these issues required looking at NLP-inspired pre-processing techniques such as sequence padding. Moreover, the use of NLP-inspired models like Transformers was shown to eliminate the need to have time as a separate input feature. 

The goal is find ways to efficiently and accurately tell whether ICU patients will survive or not given the data sampled in the past. 

The code is available [here](https://github.com/ndhers/My-Portfolio/tree/main/NLP_TimeSeries/).
