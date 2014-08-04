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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
%size(Theta1) 25*401
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%size(Theta2) 10*26

%size(X) 5000*400
             
% Setup some useful variables
m = size(X, 1);

%size(Theta1)=25*401
%size(Theta2)=10*26

% Add vector of ones
a1 = [ones(m,1) X];
z2=Theta1*a1';
a2 = sigmoid(z2);
% a2 has size 25 * 5000
a2 = [ones(m, 1) a2'];
%size(a2) 5000 *26
% a2 has size  5000 * 26
z3=Theta2*a2';
a3 = sigmoid(z3);
% a3 has size  10 * 5000 

ymatrix=y*ones(1,numel(unique(y)));

for i=1:m
ymatrix(i,:)=(ymatrix(i,:)==unique(y)');
end

% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
k=unique(y);

for i=1:max(k)
ybin = y==i;
%size(ybin)
h=a3(i,:)';
%size(h)
J = J +sum(-ybin.*log(h)-(1-ybin).*log(1-h))/m;
end

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

Delta3=a3'-ymatrix;
%size(Delta3) 5000*10

%       25*10   10*5000  

Delta2=((Theta2(:,2:end)')*(Delta3')).*(a2(:,2:end).*(1-a2(:,2:end)))';   %sigmoidGradient(a2(:,2:end)');
%size(Delta2) 25*5000
% 25*5000

D1=(Delta2*a1)/(m);
D2=(Delta3'*a2(:,:))/(m);
Theta1_grad=D1;
Theta2_grad=D2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


J= J+ lambda*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))/(2*m);

% 25 401
% 10 26
Theta1_grad(:,2:end)=Theta1_grad(:,2:end) + lambda*Theta1(:,2:end)/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end) + lambda*Theta2(:,2:end)/m;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

