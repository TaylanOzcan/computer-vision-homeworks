function [accuracy,recall,tn_rate,precision] = classifier_performance(svmClassifier,features_pos,features_neg)

% This function will return some metrics generally computed to determine the
% performance of a binary classifier. To clasify a sample first determine 
% the confidence value (also referred to as score): 
% confidence = features*w +b, 
% where features are the hog features, w is the linear classifier 
% weights and b is the classifier bias. 
% Predict a face if confidence>=0 and non-face if confidence<0

% Below you can see the definitions of true-positive, true-negative,
% false-positive and false-negative in a confusion matrix

%                  |    Predicted Non-Face |  Predicted Face
% _________________|_______________________|__________________
%  Actual Non-Face |    TRUE NEGATIVE      |   FALSE POSITIVE
% -----------------|-----------------------|------------------
%  Actual Face     |    FALSE NEGATIVE     |   TRUE POSITIVE
% -----------------|-----------------------|------------------

% You should calculate the following:
%   Accuracy: Overall, how often is the classifier correct?
%       accuracy = (TP+TN)/total = (TP+TN)/(TP+TN+FP+FN)
%   Recall: When it's actually a face, how often does it predict face?
%       recall = TP/actual_faces = TP/(TP+FN)
%   True Negative Rate: When it's actually non-face, how often does it predict non-face?
%       tn_rate = TN/actual_nonfaces = TN/(TN+FP)
%   Precision: When it predicts face, how often is it correct?
%       precision = TP/predicted_yes = TP/(TP+FP)


% remove this placeholder
%accuracy=0; recall=0; tn_rate=0; precision=0;

%===================================================================

%        YOUR CODE GOES HERE

% Initialize true/false positives/negatives
TP = 0, TN = 0, FP = 0, FN = 0;
% Get the weight and bias terms from the classifier
w = svmClassifier.weights;
b = svmClassifier.bias;
% Merge positive and negative features
features = [features_pos; features_neg];
% Calculate confidence vector using w and b
confidence = features * w + b;

% Calculate the size of positive and negative features
size_pos = size(features_pos,1);
size_neg = size(features_neg,1);

% Iterate over confidence values for positive features
for i = 1:size_pos
    % If confidence>=0, then it is a true positive
    if confidence(i) >= 0
        TP = TP + 1;
    % If confidence<0, then it is a false negative
    else
        FN = FN + 1;
    end
end

% Iterate over confidence values for negative features
for i = (size_pos+1):(size_pos+size_neg)
    % If confidence>=0, then it is a false positive
    if confidence(i) >= 0
        FP = FP + 1;
    % If confidence<0, then it is a true negative
    else
        TN = TN + 1;
    end
end

% Calculate accuracy, recall, tn_rate and precision values
accuracy = (TP+TN)/(TP+TN+FP+FN);
recall = TP/(TP+FN);
tn_rate = TN/(TN+FP);
precision = TP/(TP+FP);

%==================================================================

end