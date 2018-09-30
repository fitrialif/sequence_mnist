# sequence_mnist
Using Recurrent Neural Networks for the handwritten digit classification on MNIST dataset.  


> MNIST has become kind of default and standard choice for getting started with neural networks. But if are having this question: *"How to use mnist as sequence data and apply the RNNs on it?"*. Then you are the expected audience here.

> Moreover, if you want to see how you can code RNN model in tensorflow pretty much from scratch (that without using there inbuilt rnn api which ofcourse is more optimized), then also you can check out this code.

# UP and Running  
1. download/clone this repo and cd to the repo's root dir.  
1. ### create a virtual environment
    `python3 -m venv --system-site-packages path/to/my-venv`  
1. ### activate your virtual environment
    `source path/to/my-venv/bin/activate`  
1. ### install the dependencies
    `pip install -r requirements.txt`  
1. ### Now you are ready to run the code.
    `python3 rnn_mnist.py`  

> For any missing dependencies, install after "activating your-venv" using:  
    `pip install package-name`