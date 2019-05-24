# Machine Learning project - Image Colorization
In this project, we are focus on testing machine learning algorithms to do image colorization. Multiple models will be implemented and compared. Here are our work assignments and references:

* Hong Jiang: 
Image Colorization using CNNs and Inception-ResNet-v2; Reference: [baldassarreFe/deep-koalarization](https://github.com/baldassarreFe/deep-koalarization)

## I. Image Colorization using CNNs and Inception-ResNet-v2
### 1.Formula
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/IMG_0855.JPG)


### 2. Approach
In the original github repository, they used 130 batches, and each has 500 trainging images. However, this project requires GPU, but downloading or uploading so much training samples take considerable time, which might cause disconnect to Colab VM, and therefore need to reset runtime and would make us lose all data. Hence we only use 303 trainging samples, which means we have 100 batches, and each of them have 3 images. Because this project requires multiple .py files and functions to accomplished, we finish this lab using Ubuntu 16.04 terminal for image download, resize and transform, as well as using colab VM to train and evaluate the model. Here is our approach to do this project, the colab records can be found in file:colorization_303trainingimage.ipynb

```
cd /
```
```
git clone https://github.com/baldassarreFe/deep-koalarization
```
```
mkdir imagenet
```
```
cd imagenet # the imagenet is in root bydefault. You may change the root directory in ~/dataset/shared.py
```
```
mkdir original
```
```
mkdir resized
```
```
mkdir tfrecords
```
download the imagenet labels:
```
wget https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl
```
download the checkpoints:
```
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
```
download the image urls:
```
wget http://www.image-net.org/imagenet_data/urls/imagenet_fall11_urls.tgz
```
```
cd /deep-koalarization
```
set up a virtual environment in requirements.txt
```
python3 -m venv venv
```
```
source venv/bin/activate
```
```
pip install -r requirements.txt
```
Now we start the image downloading process:

![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/download_help.png)

```
python3 -m dataset.download -c 600 
```

the image resize process:
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/resize_help.png)
```
python3 -m dataset.resize 
```
Then we convert the images to tfrecords:
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/labbatch_help.png)
```
python3 -m dataset.lab_batch -c /imagenet/inception_resnet_v2_2016_08_30.ckpt
```
We need to prepare the valuation data before any training process. We simply take three images out of the /imagenet/original file and do the same data processing procedure. (Remeber to change the input and output file path in the command). 

Before training, ensure that the folder `~/imagenet/tfrecords/` contains:
- training records as `lab_images_*.tfrecord`
- validation records as `val_lab_images_*.tfrecord` 

Note that you need to change the number of training data, size of batch,etc in *deep-koalarization/colorization/train.py* before the training

And here is the training and evaluation record screenshots in colab. The training and valuation data were uploaded into it.
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/colab.JPG)

The result is shown as follows:
![](https://github.com/brainleft/ML-project/blob/master/CNNs%20and%20Inception-ResNet-v2/image_in_readme/result.jpg)

## Conclusion
Due to the small number of traing data, our result is kind of far from colorization. Besides, we also test CPU tensorflow for this task, and it takes forever until the colab VM lose connection. We believe that CNN + pretrain checkpoints might not be a very good idea for image colorizaiton because of the large amount of samples it requires and the computation time. 
