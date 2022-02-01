
# **Interactive 2D Fourier Analysis and filters**

This is the repository for source codes used for analysis in the final coursework challenge of ***Fourier Analysis 2*** in the winter quarter of 2021.
In this repository, you will find the interactive experience of how image filters work with the manipulation of image spectrum (Fourier Transformed 2D image).

------------

### **Table of Contents**

- [Background](#background)
- [Warning](#warning)
- [Setup](#setup)
- [Features](#features)
  - [main.py](#mainpy)
  - [fft_eraser.py](#fft_eraserpy)
- [Challenges](#challenges)
  - [1. Ideal Low Pass Filter](#challenge-1-ideal-low-pass-filter)
  - [2. Use of Filters](#challenge-2-use-of-filters)
- [Closing words](#closing-words)

------------

## **Background**
In short, after I found out that Fourier Transform can also be done to 2-dimensional images. I want to use my programming skill to make an interactive experience for us to discover and deepen our understanding of this amazing use of Fourier Transform. Therefore, I have made a program to apply image filters (which is actually convolution in color (?) domain, or multiplication in Fourier domain). Moreover, I have left some parts of the codes out, for the classmates to take the challenge and make some image filter as well.

For further background and how Math Magic works, read [the report here](link to be added).

## **Warning**
I have tried my best to ensure that the codes in this repository will work in many platforms as possible. However, due to my limited time and limited access to a platform other than Windows, I cannot ensure compatibility for other platforms. But please contact me, or open an issue in this repository, and I will try to address it as soon as possible.

**Tested platform:**
 - [x] Windows - specifically *Windows 11 Version 21H2 (OS Build 22000.469)*

Also, the performance of the program might vary from system to system. 
The interactive eraser in file [`fft_eraser.py`](./fft_eraser.py) is the one with the worst performance. But in an optimal circumstance, this file should let you remove any frequency in the spectrum with the built-in eraser like shown in the [features section](#features).

## **Setup**
1. Download this repository to your computer by clicking the green **`Code▾`** button above and `download ZIP`.
2. Extract the zip file using the program of your choice (such as 7-zip), you will get a folder that contains every file in this repository.
3. Ensure you have at least Python 3.6 installed in your system. If not, download and install Python from [the official Python website](https://www.python.org/downloads/)
4. Open the command line in the extracted folder. Then, using your preferred package manager (*`pip` in this case*), install all required libraries for this project using the command below
```Shell
pip install -r requirements.txt
```
or install them individually using this command instead
```Shell
pip install numpy matplotlib scipy
```
5. *2* python files in the topmost directory ([`fft_eraser.py`](./fft_eraser.py) and [`main.py`](./main.py)) can now be compiled and executed

## **Features**
### [main.py](./main.py)
Applying filters to the Fourier Domain of the input picture. There is slider(s) to change the parameter of the applied filter. 
This is also where you will have to do the challenges.
![main_demo](https://user-images.githubusercontent.com/67893680/151758854-de9ff272-f388-4b2b-b80d-0fb709d41b2a.gif)

### [fft_eraser.py](./fft_eraser.py)
This is only a playground (but an unstable and laggy one).
You can use the eraser to erase the part of the unwanted frequency in Fourier domain (experimental).
![fft_eraser_demo](https://user-images.githubusercontent.com/67893680/151758869-a28c844c-c8ba-4a1a-afeb-048e361d7974.gif)

## **Challenges**
Although you can choose to take the challenge immediately, it is recommended to read about what different part of the 2D FFT spectrum represents first. There are many well-explained resources in the reference of my report, the one I am recommended as a QuickStart guide is [this one](https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/). (From the *beginning* to *[The Fourier Transform and The Grating Parameters](https://thepythoncodingbook.com/2021/08/30/2d-fourier-transform-in-python-and-fourier-synthesis-of-images/#:~:text=The%20Fourier%20Transform%20and%20The%20Grating%20Parameters)* part)
### **Challenge 1: Ideal Low Pass Filter**
When you opened the [main.py](./main.py) file, you will be greeted with some parts of codes that have been commented out.
You just have to ignore it and run the file for the first time. But with your knowledge from what we have heard in our ***Fourier Analysis 2*** (Challenge number **3**), You noticed that there is one missing type of filter for you to select. (Aside from the fancy Gaussian filters)
Yes indeed, the low pass filter is missing from the filter selection panel. Don't worry, this is intentional, and also your first challenge.
#### Instruction
Using the outline code in [main.py](./main.py), complete the function `ideal_low_pass_array` to make the low pass filter work when the program runs. 
This function should take the dimension of the array (`rows` and `columns`), and the radius or `cutoff frequency`.
```Python
def ideal_low_pass_array(rows:int , columns:int, cutoff_frequency:int) -> np.ndarray:
  # Take in 3 integers, return 2-dimensional array (matrix) of shape rows × columns
  return filter_array
```

The output is expected to be a 2-dimensional array with a specified dimension. This output must also be ready to multiply *(element by element)* with the Fourier transform spectrum.
This means each member of the array **must** have values of only 0 or 1 (since this is an ideal filter).

Further instruction is provided using comments in [main.py](./main.py).
After the implementation, **don't forget to add the function to the interactive figure by using the code in the commented area**
When the filter is working in the interactive figure, try applying different values of `Cutoff frequency` (aka radius) and observe the result.

### **Challenge 2: Use of Filters**
After playing with the filters, you gained a basic idea about how filters can be used to alter the image.
#### Instruction
In simple words, discuss what can different filters do to the original image, and how they behave in that way. Moreover, think about how can this be applied further, or what can be the good application of these filters
## **Closing words**
Thank you for attempting this coding challenge. I know some of you are not familiar with or good at programming, so I have tried my best to use my programming skill to make a program to visualize this concept.
I put in my effort to make the file you are editing (maybe) easier to read/understand. I apologize in case this challenge is too hard for some people, but at least I hope you get the picture of how Fourier Transform and filter works on digital images.
