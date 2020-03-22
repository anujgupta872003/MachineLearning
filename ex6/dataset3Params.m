function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
val = [0.01 0.03 0.1 0.3 1 3 10 30];

c_sigma_pred_err = zeros((size(val)(:,2)*size(val)(:,2)),3);
c_sigma_pred_err(1:length(val),1:2)=[val',val'];
tmp = nchoosek(val,2);
c_sigma_pred_err((length(val)+1):length(val)+length(tmp),1:2)=tmp;
c_sigma_pred_err((length(val)+1)+length(tmp):length(c_sigma_pred_err),1:2)=[tmp(:,2),tmp(:,1)];



for i = 1:length(c_sigma_pred_err)
  model= svmTrain(X, y, c_sigma_pred_err(:,1,:)(i), @(x1, x2) gaussianKernel(x1, x2, c_sigma_pred_err(:,2,:)(i)));
  predict=svmPredict(model, Xval);
  
  c_sigma_pred_err(i,3) = mean(double(predict ~= yval)) ;
  
endfor
 %c_sigma_pred_err(i,3)
 
 [minval, row] = min(min(c_sigma_pred_err(:,3),[],2));

 C= c_sigma_pred_err(row,1);
 sigma= c_sigma_pred_err(row,2);
 
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
