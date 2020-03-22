function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];

a1 = X;
Z2 = a1*Theta1';
a2 = sigmoid(Z2);
a2 = [ones(size(a2,1), 1) a2];
Z3 = a2*Theta2';
a3 = sigmoid(Z3);
h = a3;
I = eye(num_labels);
Y = zeros(size(y,1),num_labels);


Y=I(y, :);



  J = -(1/m)*(sum(sum( ((Y.*log(h))+((1-Y).*log(1-h)))))) ;
  reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
  
  J = J + reg;
Tdelta1 = zeros(size(Theta1));
Tdelta2 = zeros(size(Theta2));

for i = 1:m
  
xback = X(i,:);
yback = Y(i,:);
af1 = xback;
Z2_f = af1*Theta1';
af2 = sigmoid(Z2_f);
af2 = [ones(size(af2,1), 1) af2];
%af2 = [1; af2];
Z3_f=af2*Theta2';
af3 = sigmoid(Z3_f);
delta3 = af3 - yback;  

%Z2_f = [1 ; Z2_f];
tt = Theta2'* delta3';
delta2 = tt(2:end,:)' .* sigmoidGradient(Z2_f);

Tdelta2 = Tdelta2+ delta3'*af2;
Tdelta1 = Tdelta1+ delta2'*af1;
endfor

if J == 0 then
  Theta1_grad = (1/m)*Tdelta1;
  Theta2_grad = (1/m)*Tdelta2
else
  Theta1_grad = (1/m)*Tdelta1 + (lambda/m)*[zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
 Theta2_grad = (1/m)*Tdelta2 + (lambda/m)*[zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
endif


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
