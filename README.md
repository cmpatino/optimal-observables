# Optimal Observables

The objective of this project is to use neural networks and symbolic regression to find a closed-form expression of a classifier that separates signal and noise on particle physics problems. We call this closed-form expression an optimal observable since it allows us to separate signal and noise optimally using quantities we can observe experimentally.

Usually, there are two ways of designing an optimal observable that separates signal and noise. The first method is a subject matter expert who chooses a few relevant quantities we can observe experimentally. The expert then applies transformations to those observables, resulting in a good separation between signal and noise. This expert-based method happens in a few dimensions since it is difficult for humans to visualize the effects of feature transformations on higher dimensions.

The second method uses machine learning-based methods. Machine learning models, and specifically neural networks, can process high-dimensional data to achieve a good separation between two classes. However, neural networks lack the closed form and interpretable expressions that result from observables designed by subject matter experts.

Symbolic regression can bridge the gap between closed-form expressions and neural networks to generate observables optimized in high-dimensional spaces that we then approximate with an analytical expression. Bridging this gap allows us to learn optimal observables through the power of machine learning while being able to extract physical meaning from the function learned by the model.

**This is a work in progress.**

## Tech Stack
+ [JAX](https://github.com/google/jax)
+ [PyTorch Lightning](https://github.com/Lightning-AI/lightning)
+ [PySR](https://github.com/MilesCranmer/PySR)
+ [Awkward Array](https://github.com/scikit-hep/awkward)
+ [uproot](https://github.com/scikit-hep/uproot5)
+ numpy
+ matplotlib

## Project Details

The setup we have for modeling is a neural network with inputs $X$ that learns to classify whether a sample comes from signal or noise. We set $X$ to quantities we can observe experimentally so that the neural network learns a representation that maximizes the separation between signal and noise using only quantities we can observe.

The optimal observables we use want to learn have a multiplicative form $o = \prod \tau^{\alpha}$, which can be written as $\log(o) = \sum \alpha \log(\tau)$. The last expression is equivalent to the last layer of a neural network where $\log(\tau)$ are the representations learned by the network on the second to last layer. $\alpha$ are the weights on the last layer, and $o$ is the output neuron before applying the sigmoid function. Since $o$ is the output for the classifier, we train it to maximize separation between the classes corresponding to noise and signal. The training then guarantees that $o$ is an observable that can separate signal and noise. Below is a diagram of the overall setup.

![architecture_diagram](./docs/images/project_diagram.png)

We use symbolic regression to generate a closed-form expression of the representations $\log(\tau)$ learned by the neural network. Specifically, the neural network learns a function $\log(\tau) = f(X; w)$ where $w$ are the weights of the neural network. Through symbolic regression, we approximate $f(X; w) \approx g(X)$ where we allow $g(X)$ to be a function of sums or products of the input quantities. The approximation allows us to write the function learned by the neural network in a closed form and give a physical interpretation of the result.

By having an analytic approximation, we have the best of both worlds: we learn an optimal observable through optimization on a high-dimensional space while being able to extract physical knowledge from a closed-form expression. The objective of this project is to use neural networks and symbolic regression to find a closed-form expression of a classifier that separates signal and noise on particle physics problems. We call this closed-form expression an optimal observable since it allows us to separate signal and noise optimally using quantities we can observe experimentally.