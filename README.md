Crossnet, a cross-view supervised learning solution
========

This code is a tensorflow implementation of this paper, [Predicting Ground-Level Scene Layout from Aerial Imagery](https://arxiv.org/pdf/1612.02709.pdf).

And you are welcome to visit our [Project Page](http://cs.uky.edu/~ted/research/crossview/)

Dependencies
------------
* Tensorflow r1.1 or higher [Installation Page](https://www.tensorflow.org/versions/r1.1/install/).

Running
------------
Training:
```bash
$ python main.py
```

Deploying:
```bash
$ python main.py --is_training=False
```
Data
------------
This repo only contains some example data for training, the whole dataset can be found in [this link](https://drive.google.com/open?id=0BzvmHzyo_zCAX3I4VG1mWnhmcGc).
You would have to edit the dataset path in main.py in order to use it.

Note
------------
We plan to release evaluation code soon.
