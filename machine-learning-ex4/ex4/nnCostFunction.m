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
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));
D3 = zeros(m,num_labels);
D2 = zeros(m,hidden_layer_size)

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%Input Layer
a1 = [ones(size(X,1),1) X];%(m*n+1)

%Hidden Layer
z2 = a1*Theta1';%(m*25)
a2temp = sigmoid(z2);
a2 = [ones(size(z2,1),1), a2temp];

%Output Layer
z3 = a2*Theta2';
a3 = sigmoid(z3);
p = a3;


Y = y;
for (i = 1:m)
Yvect = 1:num_labels;
Yvect = Yvect ==y(i,:);
P = p(i,:);
J = J+(1/m.*sum((-Yvect*log(P)')-((1-Yvect)*log(1.-P)')));
end

Theta1temp = [zeros(size(Theta1,1),1),Theta1(:,2:(input_layer_size+1))];
Theta2temp = [zeros(size(Theta2,1),1),Theta2(:,2:(hidden_layer_size+1))];

J = J+lambda/(2*m)*(sum(sum(Theta1temp.^2)) + sum(sum(Theta2temp.^2)));

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
for (t = 1:m)
  %Step 1
    %Perform Forward Propagation
      %Input Layer
      a1 = X(t, :);
      a1 = [1,a1];
      %Hidden Layer
      z2 = a1*Theta1';%(1*hidden_layer_size)
      a2temp = sigmoid(z2);
      a2 = [ones(size(z2,1),1), a2temp];%(1*hidden_layer_size+1)      
      %Output Layer
      z3 = a2*Theta2';%(1*num_labels)
      a3 = sigmoid(z3);
      p = a3;
  %Step 2
    y_array = 1:num_labels;
    y_array = y_array == y(t,:);
    d3 = p-y_array;%(1*num_labels)
  %Step3
    d2 = d3*Theta2(:,2:end).*sigmoidGradient(z2);%(hidden_layer_size+1*1)
  %Step4
    D3(t,:) = d3;
    D2(t,:) = d2;
    Delta2 = Delta2.+(d3'*a2);
    Delta1 = Delta1.+d2'*a1;
    %Theta1_grad = Theta1_grad+Delta2'*a1;
end;
  %Step5
  Theta2_grad = Delta2/m;
  Theta1_grad = Delta1/m;
  
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
