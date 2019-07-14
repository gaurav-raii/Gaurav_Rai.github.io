# Portfolio_Gaurav_Rai
Project Portfolio

This is a repository of the projects I worked on or currently working on. It is updated regularly. The projects are either written in R  or Python (Jupyter Notebook or .py format). The goal of these projects is to use data science/statistical modelling techniques to draw interesting and useful insights from different publicly available datasets. A typical project consist of finding and cleaning data,visualization, analysis and conclusion. Click on the projects to see full analysis and code. 

More information about me: [LinkedIn](https://www.linkedin.com/in/gaurav-r-rai/)

My Email: gauravrai@tamu.edu

## Projects

### Topic-analysis(NLP) to assess the level of damage during a car crash.( Course Project for Applied Analytics Class)

* Developed a model to assess the damage to a vehicle during an accident using the text data from 
complaints submitted to National Highway safety and Traffic Administration( NHSTA).
* The data contained 2,375 complaints about specific GMC vehicles submitted to the NHTSA. The complaints were under column description along with several other variables such as make, Model, year, mileage of the vehicle.
* Conducted a text classification analysis on this data ,using POS tagging , Stop_words removal, stemming 
for building the term/doc matrix.
* Used TF-IDF weights for Term/Doc Matrix.
* 8 Clusters of complaints were formed. Probability of an observation falling into one of these topics 
Clusters(T1 to T8) along with make, model, year, mileage were used to classify the level of damage into 
low, medium and high.
* This type of models can be used to automatically prioritize and route the resources based on 
information recieved in the form of texts. emails etc in emergency scenarios such as a wild fires and 
natural disasters where saving even seconds in taking actions can make all the difference.

 Tools Used: Python

Packages: NLKT, Pandas

### Customer segmentation using unsupervised learning methods

* The idea behind customer segmentation for businesses is finding the behavioral patterns in customers for that particular business.
* These insights in-turn can really come in handy for targeted marketing and also increasing business productivity.
* The techniques used were:
1) K-means clustering algorithm
2)Mean Shift Clustering
3) Hierarchical clustering algorithm

Tools used: Python

Packages used: sk-learn, PCA

### Trigger Word Detection

* Built a trigger word detection system which detects the word "activate" from speech using a Recurrent Neural Network.
* Generated artificial speech training data by inserting positive and negative words snippets on different 10 seconds long background noise clips.
* Computed the spectograms for each training observation
* used a 1-D Convolution layer and two GRU layers with Batch Normalization and Dropout (rate= 0.8).

Framework used: Keras
