---
layout: splash
classes:
  - wide
---

# Practically Speaking: Tensorflow 2 vs PyTorch
*A quick primer for the key similarities and differences between Tensorflow 2 and PyTorch* <br>

### *Jacob Sieber*
### *Febuary 1, 2019*

<a><img src="https://i.imgur.com/0Fc4jsB.png" /></a>

If you are looking to build neural networks, Google's Tensorflow 2 and Facebook's PyTorch are the two most fleshed out tools to quickly go from zero to state of the art. Alongside the newest neural network papers, you will often find an implementation of the neural network written in one of these two frameworks. The main implementation of these deep learning frameworks are used via the Python language. Interesetingly, I haven't been able to find a good comparison inbetween the most recent versions of these extremely popular Python packages online, so this primer will help you understand the key similarites and differences inbetween Tensorflow 2 and Pytorch.


```python
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

import torch
import torch.nn as nn

print('Tensorflow version:', tf.__version__)
print('PyTorch version', torch.__version__)
```

    Tensorflow version: 2.0.0
    PyTorch version 1.4.0


## Tensorflow 2.0 - The All Inclusive Deep Learning Framework

Tensorflow has been built completely from the ground up to provide industrial speed while still allowing for deep customizated deep learning models. Google has built every aspect from streaming preprocessing dataloaders to Tensorboard model visualization to work in perfect harmony. The key benefits of Tensorflow 2.0 compared to PyTorch are listed below.  

### Training Speed

Tensorflow speed, particularly when eagerly execution is turned off, is a what many drives many large scale problem solvers to Tensorflow. In some cases, Tensorflow can train up to 10% faster than PyTorch in very computational heavy environments. Training speed is often a make or break decision in environments with low computational power (i.e. mobile devices) and areas where the model is always being trained (such as recommender systems for large internet-based companies). Tensorflow can also take advantage of Tensor Processing Units (TPUs) to a much greater extent. However, as we look into some of PyTorch's strengths, we will find a trade off inbetween speed and flexibility. 


```python
print('Executing Eagerly?', tf.executing_eagerly())
tf.compat.v1.disable_eager_execution()
print('Executing Eagerly?', tf.executing_eagerly())
tf.compat.v1.enable_eager_execution()
```

    Executing Eagerly? True
    Executing Eagerly? False


### Integrated Tools, Levels of Abstraction, and Ease of Use

Many of Tensorflow's 2.0 Python upgrades incorporated vast amounts of feedback from users about the importance of tools that make experimentation simpler and more maintainable. Perhaps the biggest upgrade for Tensorflow 2.0 was eager execution - allowing for a dymanic graph. Tensorflow developers did not stop there, and added a bunch of features that made it much easier to quickly build state of the art neural networks no matter what level of abstraction you wanted to go in. Below are some of the awesome tools built in.

#### Abstractions when building Neural Networks

Unlike PyTorch, Tensorflow has three main levels of abstraction that cuts down on boilerplate code. They are called the Sequential API, the Functional API, and the Model API. You can use the fit method on a Tensorflow model to train, or you can build a custom training loop. I would however be remiss to not mention that PyTorch does have some third party libraries that offer similar functionality, but as well integrated as Keras and Tensorflow. PyTorch also has a sequential function, but does not have the same simple fit method.


```python
# Sequential API
# From https://www.tensorflow.org/tutorials/keras/classification

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2)
```

    Train on 60000 samples
    Epoch 1/2
    60000/60000 [==============================] - 2s 36us/sample - loss: 0.4922 - accuracy: 0.8267
    Epoch 2/2
    60000/60000 [==============================] - 2s 33us/sample - loss: 0.3725 - accuracy: 0.8654


##### Simple and Intergrated Tools: Callbacks
Callbacks are a great example of some of Tensorflow's excellent integrated tools. I find that TF 2.0 callbacks are a fair more intuitive compared to PyTorch callbacks. When calling the train method of a nerual network model, you can simply feed a list of callbacks to the model as an argument. Here are two callbacks that I use most commonly in my Tensorflow projects.


```python
# The one_cycle_sched is code adapted from Jeremy Howard's 2019 Fastai Course
# This callback the base LearningRateScheduler class from Tensorflow

from functools import partial
import math
import numpy as np

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, Iterable): return list(o)
    return [o]

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2

def combine_scheds(pcts, scheds):
    """
    ::pcts:: indicates what % through the function we are
    ::scheds:: the functions involving LR
    """
  
    assert sum(pcts) == 1., "Needs to cover 100% of the cycle."
    pcts = np.array([0] + listify(pcts))
    assert all(pcts >= 0), "Can't have negative % of cycle."
    pcts = np.cumsum(pcts, 0)
    
    def _inner(pos):
        """
        Function that we can insert the position (0-100%) of the model and get the proper
        learning rate.
        """
        idx = np.array((pos >= pcts).nonzero())
        idx = idx.flatten()[-1]
        try:
            actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        except IndexError:
            idx -= 1
            actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)    

    return _inner

def one_cycle_sched(max_lr, total_epochs, verbose=0):
    """
    Returns a callback function for fit one cycle LR adjustment.
    """
    one_cycle_sched = combine_scheds([0.3, 0.7], [sched_cos(max_lr/2, max_lr), sched_cos(max_lr, max_lr/3)])

    def scheduler(epoch):
        progress = epoch / total_epochs
        lr = one_cycle_sched(progress)
        return lr

    return keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose)

history = model.fit(train_images,
                    train_labels,
                    epochs=4,
                    callbacks=[one_cycle_sched(1e-3, 4)],
                    verbose=0)

print('Learning Rates:', history.history['lr'])
```

    Learning Rates: [0.0005, 0.00096650637, 0.0008744966, 0.0005220387]



```python
# This example inherits from the base callback class to print progress to
# print to console on every every epoch_end.
# It's helpful when you have a nerual network that has very quick epochs

import sys

class cb_simple_progress(keras.callbacks.Callback):    
    def __init__(self, epochs, total_notches = 36):
        self.epochs = epochs   # epochs need to be provided in advance
        self.total_notches = total_notches

    def on_epoch_end(self,epoch,logs=None):
        loss = int(logs['loss'])
        if logs.get("val_loss"):
            val_loss = f" Valid Loss: {int(logs.get('val_loss')):,}"
        else:
            val_loss = ""

        cur_notches = int((epoch / self.epochs) * self.total_notches) + 1
        progbar = "[" + "=" * cur_notches + "." * (self.total_notches - cur_notches) + "]"
        out_str = ('Epoch %d/%d ' % (epoch + 1, self.epochs) + progbar +
          " Training Loss: {:,}".format(loss) + val_loss)
        sys.stdout.write('\b' * 200)   # Erases old output
        sys.stdout.write(out_str)
```

### Packages maintained by Google and Functionality beyond Python

Rather than heavily rely on third party packages, Tensorflow builds out most of its external functionailty and libraries. The most popular example of an external package is Tensorboard, the nerual network visualization tool which even PyTorch is now integrated with. The most interesting of these tools are beyond the scope of Python. Some of the key initivates are building production ready piplelines, working on embedded devices & low power devices, and working with Swift to have a deep learning framework on an expressive language with first class auto differentiation. This is allows a single framework to be used from experimentation to large scale training to deployment.

<img src="https://i.imgur.com/Ot2PjeJ.png"/>

## PyTorch - The Flexible Deep Learning Framework

Originally built for researchers, PyTorch is a deep learning framework that allows for Pythonic development where Tensors on the GPU can be handled in almost the same way as a Numpy Array. In research, PyTorch has historically been prefered over Tensorflow due to the ease of builing highly customized neural networks and the ability to tweak every aspect of neural networks at a low level while still retaining relatively clean and simple code. The key benefits of PyTorch compared to TensorFlow are listed below.

### Consistent Pythonic Implementation

Instead of imposing its own rules, PyTorch follows Pythoic best practices in terms of implementation. This means you can build neural networks, code your training loops With the simplicity of Python . A good example of this is how to run a neural network on parallel GPUs. PyTorch makes this seemingly difficult task a breeze.


```python
# Parallel of a Nerual Network in Pytorch
# Used my cpu as an substitute second GPU - I only have 1 GPU
# From: https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html

import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cpu')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cpu'))
    
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cpu')
loss_fn(outputs, labels).backward()
optimizer.step()
```

*Training Loop Example from Deep Reinforcement Learning Hands-On Second Edition by Maxim Lapan* <img src="https://i.imgur.com/F3EgTDL.png"/>

### Manipulating Tensors and Gradients

The developers of PyTorch have gone a great length to ensure that conducting experiments and building neural networks from the ground up is as simple as possible. When reviewing other's code, the consistency and simplicity of PyTorch code helps out a great deal. Most of the time, manipulating tensors and gradients is as simple as dealing with Numpy arrays. It is hard to encapsulate the many times this comes in handy, but here is another example that you may run into quite often.


```python
# PyTorch tensors support direct assignment

tf_tensor = tf.convert_to_tensor(range(5))
torch_tensor = torch.tensor(range(5))

try:
    tf_tensor[3] = 10
except:
    print("TypeError: Tensorflow Eager Tensors "
          "do not support item assignment")


torch_tensor[3] = 10
print('Torch Tensor sucessfully assigned')
```

    TypeError: Tensorflow Eager Tensors do not support item assignment
    Torch Tensor sucessfully assigned


### Extensive Cuting Edge Research

Because of PyTorch's Pythonic implementation and ease of use, PyTorch is popular with researchers. The great thing about the ai / deeplearning revolution is that most of the research is opensourced and also have github implementations. You can reuse this code (with proper citations) often directly. For example, the winning m4 forecasting algorithm is on [github](https://github.com/damitkwr/ESRNN-GPU) ready for download. PyTorch is especially suited for coding up research papers and seeing the results yourself.



## The Final Word (so which should I use)?

Like most articles that compare two different pieces of software, I will say that it depends on your use case. I'll list when I think that each toolkit is best - but keep in mind this is a highly opinionated topic and I'm sure that I will recieve a few scathing emails.

### I'm just starting to get into neural networks - which framework will be the quickest to build cool neural networks?

I would recommend Tensorflow 2.0. If you are looking for in depth learing material, I also highly recommend [Aurelien Geron's Hands on Machine Learning](https://www.amazon.co.uk/Hands-Machine-Learning-Scikit-Learn-TensorFlow/dp/1492032646). If you also want to consider higher level Python packages - Fastai built on Pytorch is an excellent starting point (and models are often state of the art).
* Tensorflow has superb documentation that walks you through problems that can teach you the ropes with simplicity and minimal code. It cannot be emphasized enough how important good documentation is when learning a vast library from scratch. 
* When using the Keras Sequential Api (PyTorch also has a similar Sequential function), you can build an excellent model, from DataFrame to predictions, in a minimal amount of code. You can slowly go to deeper levels of neural networks (such as explitly coded training loops) when needed.
* There are great examples online, particulary on Kaggle, of simple yet highly accurate neural network models.
* As you gain more experience building Tensorflow models, you can transition over to the fine-grained control of PyTorch. 

### I'm a developer that is less concerned about experimentation and more about efficently building a good enough neural network and deploy it via JavaScript and mobile devices.

I would recommend Tensorflow 2.0
* You can build models in the Keras API with minimal code. 
* Tensorflow has very robust JavaScript and mobile device support.
* Able to switch to a static graph for better speed on large datasets with optimized CUDA implementations.

### I want to code from the ground up and adapt a neural network implementation to my use case - such has reinforcement learning with Open AI Gym or coding up something I saw in a research paper.

I would recommend PyTorch

* PyTorch is Pythonic and handling tensors is intuitive. This means that you can experiment with customized solutions quickly and incorporate third party packages in a simple manner with the flexibilty of PyTorch.
* If you look at research paper repositories on GitHub, you'll find that the majority of recent research papers are coded in PyTorch. There is a good chance that a researcher worked on a similar problem as yours, you can even check websites like www.paperswithcode.com to find state of the art neural networks and use the code yourself.

## Thanks for Reading!
