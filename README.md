Thinq 

A simple neural network made for education purposes. <br/>
The name due to it's artificial intelligence origin "Think" combined with an fluent API reminding me of LINQ<br/>
Thus Think + Linq = Thinq

Background:

Deep learning is a hot topic at this moment but non the less it seems even the most fundamental understanding<br/>
how it really works is very poorly understood, as is what it's suitable for and what it's strengths and weaknesses are.

This is my attempt to implement a fundamental feed forward network in the simplest possible way, <br/>
both in terms of implementation but also in terms of API. 



API:

Thinq consists of basically two components, the Neuron layer class and the Neural Network class.<br/>

The neuron layer represents a set of inputs and neurons, the number of inputs and neurons are always symmetrical,
the outputs are automatically scaled depending on the number of inputs of the previous layer. 

A neural network generally consist of an input layer, a hidden layer and an output layer<br/>

See the examples for further information on the actual API (which is still work in progress).<br/>

Thinq has experimental support for different activation functions<br/>

* ReLU - Rectified Linear Unit 
* tanh - Hyperbolic Tangent
* softmax - Softmax probability distribution
* sigmoid - Traditional sigmoid 

** 

To create neural network with Thinq <br/><br/>

var network = ThinqNeuralNetwork(new ThinqNeuronLayer('Name of the input layer', number of inputs),
                                 new ThinqNeuronLayer('Name of the hidden layer', number of neurons, activiation function),
                                 new ThinqNeuronLayer('Name of the output layer, number of outputs, activiation function));
                                 
Now you have an untrained network, to train the network you need to feed it data as well as expected output. <br/>
So your instance called "network" in this example, has a method called fit- which will attempt to fit the data using 
gradient descent, you may need to iterate fit with different data over multiple passes until the network converge in a pleasing way.<br/>
<br/>
Once trained you can call "predict" with associated input data- and you'll get a response from the network with the output (depending on activiation function choosen)
<br/>
Additionally if you want to persist/restore a trained network you may use loadANN or saveANN respectively which returns an object structure 
that you may persist using a suitable technique. 

<br/><br/>

                