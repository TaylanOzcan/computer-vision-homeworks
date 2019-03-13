function svmClassifier = svm_training(features_pos, features_neg)
% This function will train your SVM classifier using the positive
% examples (features_pos) and negative examples (features_neg).

% Use label +1 for postive examples and label -1 for negative examples

% INPUT:
% . features_pos: a N1 by D matrix where N1 is the number of faces and D
%   is the hog feature dimensionality
% . features_neg: a N2 by D matrix where N2 is the number of non-faces and D
%   is the hog feature dimensionality

% OUTPUT:
% svmClassifier: A struct with two fields, 'weights' and 'bias' that are
%       the parameters of a linear classifier

% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b'
% [w b] = vl_svmtrain(X, Y, lambda) 
% http://www.vlfeat.org/sandbox/matlab/vl_svmtrain.html
% 'lambda' is an important parameter, try many values. Small values seem to
% work best e.g. 0.00001, but you can try other values

%placeholders to delete
%w = rand(size(features_pos,2),1); %placeholder, delete
%b = rand(1); 


lambda = 0.00001;
%===================================================================

%        YOUR CODE GOES HERE

% Merge positive and negative features
features = [features_pos ; features_neg];
features = transpose(features);
% Form the vectors for positive and negative labels
labels_pos = ones(size(features_pos, 1), 1);
labels_neg = -1 * ones(size(features_neg, 1), 1);
% Merge the positive and negative labels
labels = [labels_pos; labels_neg];
% Train the classifier to get weight and bias
[w, b] = vl_svmtrain(features, labels, lambda);

%==================================================================


svmClassifier = struct('weights',w,'bias',b);
end