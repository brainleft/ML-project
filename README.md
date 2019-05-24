# Machine Learning project - Image Colorization
In this project, we are focus on testing machine learning algorithms to do image colorization. Multiple models will be implemented and compared. Here are our work assignments and references:

* Hong Jiang: 
Image Colorization using CNNs and Inception-ResNet-v2; Reference: [baldassarreFe/deep-koalarization](https://github.com/baldassarreFe/deep-koalarization)

## I. Image Colorization using CNNs and Inception-ResNet-v2
### 1.Formula
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/IMG_0855.JPG)


### 2. Approach
In the original github repository, they used 130 batches, and each has 500 trainging images. However, this project requires GPU, but downloading or uploading so much training samples take considerable time, which might cause disconnect to Colab VM, and therefore need to reset runtime. Here we only use 303 trainging samples, which means we have 100 batches, and each of them have  
