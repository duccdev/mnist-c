# Why?

I randomly thought of this a few days ago. It sounded funny. I had to implement it.

# How do I run it?

Well, I didn't properly make model saving/loading so it just trains on start.  
You will need to download `mnist.npz` and put it inside the `py` folder, then run `python3 convert.py`.  
You will need to have `numpy` installed.  
Usually I download it from `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`.  
After running the conversion script, put the resulting file (`mnist.bin`) into `../datasets`.  
Finally, you can `cd .. && make && bin/mnist-c`.  
It doesn't have an interactive CLI, all it does is train and evaluate the model on unseen data.

# Misc

The `py` folder contains a reference implementation of the model using Python and NumPy.  
It also contains a script for converting `mnist.npz` to my own binary format.
