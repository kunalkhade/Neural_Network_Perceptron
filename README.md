# Neural_Network_Perceptron
Neural_Network_Perceptron

## Description 
Develop generic binary classifier perceptron class in ML.py. It has to take
training set of any size. Class must include four functions : init (), fit() ,
net input(), predict(), One more supportive function to display result.

Terminology :
w = Weights
s = Inputs
b = Bias
g = Activation Function
Y = Expected Output
Ŷ i = Actual Output
α = Learning Rate

Modeling Steps :
1. Linear Model : f (w, b) = w T x + b
2. Activation Function : Unit Step Function
(
1 if z ≤ 1
g(z) =
0 Otherwise
3. Approximation : y = g(f (w, b)) = g(w T x + b)
4. Perception Update Rule : w = w + 4w
4w = α(Y i − Ŷ i)
Update Rule : Weights are pushed towards positive or target class in case of
miss-classification !
