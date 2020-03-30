
### Assignment 5 

#### Question 1: 

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

*Solution:* The module has been achieved in the python file, in which I commented for part 1

**Part 2:** Now repeat the process for 1, 000 images from the test set. Plot the average accuracy
of the model on the 1, 000 adversarial examples x
∗ you produce as a function of ε. That is, you
should produce a graph with the model’s accuracy on the vertical axis and ε on the horizontal
axis. Also hand in the python code used to generate this graph (you should use the matplotlib
library to generate the plot).

*Solution:* The module has been achieved in the python file, in which I commented for part 2

**Part 3:** We will now refine the way the perturbation is found by adding several smaller
perturbations to the image rather than modifying the image in one large perturbation. Modify
the code you wrote in the first question to instead iteratively perturb the input as follows. For
k iterations, take the input x, compute x
∗ = x + ε · sign(
∂L
∂x ), replace x by x
∗ and repeat.
Take the same test image than in the first question and show that you can find a misclassified
perturbed image for k = 5 and a smaller value of the hyperparameter ≈
ε
k
. You are expected to
hand in (1) the code used to find the perturbation, (2) a visualization of the perturbed image,
and (3) the prediction vector output by the model on the original and perturbed image.

*Solution:* The module has been achieved in the python file, in which I commented for Part 3 

**Part 4:** For k = 5, plot the same graph than in the second question

*Solution:* The module has been achieved in the python file, in which I commented for part 4.
  
