# Threat-Detection

In public places, especially the ones with less population, the attention for any threat that happens is very less.
So the basic idea is to create an IOT device that captures an image once in a regular interval and sends it to a centralized server, and hence classifies it into 
   
1 - normal 

   2 - fire

   3 - theft

   4 - accident


The image classification code is given here. It uses VGG16 CNN model.
I took the images for theft and accidents from UCF crime dataset (vandalism, RoadAccidents) and for fire, I used different sources.

Dataset I used :

https://drive.google.com/drive/folders/1jIPBKXk_n3OHwYnqos5j9TrnKuaPn1Vb?usp=sharing
