
>[!idea] 
>We identify the objects being generated as vector $z \in R^d$  

The task of generating objects that are represented as vectors $z \in R^d$ as images, videos, and molecular structures.

>[!idea]
>Generating an object z is modeled as sampling form the data distribution $z ~ p_{data}$

There are different photos to represent a dog, the probability distribution to generate a dog is $p_{data}$.

Also can understand as a PDF (probability density function) it gives any object (vector z represent a photo) a likelihood greater than 0, $p(z) \ge 0$ 

Generation is the task of generating samples from a probability distribution $p_{data}$ having access to a dataset of samples $z_1, . . . , z_N ∼ p_{data}$ during training. (~ means sample)

>[!idea]
>A dataset consists of a finite number of samples $z_1, . . . , z_N ∼ p_{data}$

>[!idea]
>Guided generation involves sampling from $z ∼ p_{data}(·|y)$, where y is a conditioning variable.

Guided generation assumes that we condition the distribution on a label y and we want to sample from $p_{data}(·|y)$ having access to data set of pairs $(z_1, y) . . . , (z_N , y)$ during training.

# Flow Model

for every time t and location x we get a vector $u_t(x) \in R^d$ specifying a velocity in space