function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hX = sigmoid(X * theta);
[cost, gd] = costFunction(theta, X, y); %reuse previous function

J = cost + lambda/(2*m) * (sum(theta.^2)-theta(1)^2)
grad = 1/m * X' * (hX - y) + lambda/m * theta;
grad(1) = 1/m * sum((hX - y) .* X(:,1));


% error when grad = 1/m * sum((hX - y) .* X);+ lambda/m * theta;
% -> it turns out to be const + vector
% -> all elements in the vector have same value
%
% grad is (1 x 28) vector, X is (m x 28) vector
% =============================================================

end
