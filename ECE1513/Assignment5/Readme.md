
#### Assignment 5 

##### Question 1: 

**Part 1**: In this question, you will search for an adversarial example. Let x be an image
from the test set which is correctly classified by the model. To obtain an adversarial example,
you should modify x into x
∗ = x + ε · sign(
∂L
∂x ) where L is the cross-entropy loss and ε is a
hyperparameter. You can set the hyperparameter to a value around ε = 0.3 for the purpose of
this question. Using JAX, find a perturbed image x
∗ which is misclassified by the model while
the original image x was originally correctly classified. For the original image, take an image
of the class 7 from the test set. You are expected to hand in (1) the code used to find the
perturbation, (2) a visualization of the perturbed image, and (3) the prediction vector output
by the model on the original and perturbed image

  
