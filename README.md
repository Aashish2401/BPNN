# BPNN
An assignment problem to recognize 6 handwritten characters from an image using a simple back propagation neural network.

1. The Training_Model.py is run first in order to train the Back Propogation Neural Network using the EMNIST dataset.
2. After running the program, the configuration of the models are stored as 2 '.h5' files. One for letters and one for digits.
3. After this process is completed, the Image_Recog.py is run. 
4. This program takes the image of the text to be recognized and splits it into 5 images for each character.
5. It then imports the models trained before and passes each character through it.
6. The recognized characters are stored in a string and displayed as output to the user.

The user must remember to enter correct file path for the EMNIST dataset in Training_Model.py and for the test image, and trained models
in Image_Recog.py.
Also Python 3.6 is only supported.
Modules required:
1. Tensorflow
2. Keras
3. OpenCV
4. Numpy
5. Pillow
