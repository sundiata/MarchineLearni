# Football Stars Image Classifier

#### This is a Machine-Learning Model which classifies between 10 football players, whose names have been listed in the form of a Dictionary. The main file is located in server folder by the name 'app.py' which is a Streamlit driven python file, which runs on Localhost. 

### I have attached a screenshot of the final API.

![Screenshot (66)](https://user-images.githubusercontent.com/87764530/173772990-d0ee68d1-0948-410e-becd-797f13218ccc.png)


The project is divided into 3 parts:

#### 1. Converting Data into usable format and data cleaning<br>
          OpenCV has been used for this purpose majorly. Image classification in this case uses only face as the parameter for checking. So first all the images are 
          cropped into images which contain only face provided both the eyes are visible (using Haarcascades).

#### 2. Training the model<br>
          Model has been created on the concept of SVMs with Linear Kernel (polynomial kernel with degree = 1) and achieving a maximum accuracy of 75.3 %

#### 3. Executing the saved model<br>
          Artifact loader, wavelet transform and main function - classify_image are used to execute the joblib file (saved format of trained model). The result is the 
          predicted player(s)' names and the input image.
          
          
