function [features_pos, features_neg] = get_training_features...
    (train_path_pos, train_path_neg,hog_template_size,hog_cell_size)

%This function returns the hog features of all positive examples and for
%negative examples

% INPUT:
%   . train_path_pos: a string which is path to the directory containing
%        36x36 images of faces (Positive Examples)
%   . train_path_neg: a string which is path to directory containing images
%       which have no faces in them. (Negative Examples)
%   . hog_template_size: size of hog template. In this function it will be 
%       equal to the image size (i.e. 36) 
%   . hog_cell_size: the number of pixels in each HoG cell. Image size 
%       should be evenly divisible by hog_cell_size.

%     Smaller HoG cell sizes tend to work better, but they make things
%     slower because the feature dimensionality increases and more
%     importantly the step size of the classifier decreases at test time.

% OUTPUT
% . features_pos: a N1 by D matrix where N1 is the number of faces and D
%       is the hog feature dimensionality, which would be
%       (hog_template_size / hog_cell_size)^2 * 31
%       if you're using the default vl_hog parameters
% . features_neg: a N2 by D matrix where N2 is the number of non-faces and D
%       is the hog feature dimensionality

% Useful functions
% vl_hog()
% rgb2gray()

%% Step 1: Determine features for positive images (face images)
% This part should create hog features for all positive training examples 
% (faces) from 36x36 images in 'train_path_pos'. 

% Each face should be converted into a hog grid according to 
% 'hog_cell_size'. For example a hog_cell_size of 6 means there are 6x6 
% pixels in one cell. The hog grid will be of size 6x6 for images of size
% 36x36. A hog vector of length 31 will be computed for each cell.

% For improved performance, try mirroring or warping the positive 
% training examples.
image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
num_images = length(image_files);


% placeholder to be deleted
% features_pos = rand(num_images, (hog_template_size / hog_cell_size)^2 * 31);

%===================================================================

%        YOUR CODE GOES HERE

% Compute the size of hog feature
feature_size = (hog_template_size/hog_cell_size)^2 * 31;
% Initialize positive feature matrix
features_pos = zeros(num_images,feature_size);

% Iterate over the positive images to compute hog features for each
for i = 1:num_images
    % Read the image
    image = imread(fullfile(image_files(i).folder, image_files(i).name));
    % Convert to single
    image_single = im2single(image);
    % Compute hog
    hog = vl_hog(image_single, hog_cell_size);
    % Get 1xFEATURE_SIZE feature vector
    feature_vector = reshape(hog, [1, feature_size]);
    % Place the vector into the relevant row in the positive features matrix
    features_pos(i, :) = feature_vector;
end

%==================================================================
    
%% Step 2: Mine Negative Samples (non-face) and determine hog features

% This part should return hog features from negative training examples 
% (non-faces). Higher number of negative samples will improve results
% however start with 10000 negative samples for debugging

% Images should be converted to grayscale, because the positive training 
% data is only available in grayscale. 

% The set of non-face images available in the dataset vary in size.
% (unlike face images which were all 36x36). You need to mine negative samples
% by randomly selecting patches of size hog_template_size. This ensures the feature
% length of negative samples matches with positive samples. you might try 
% to sample some number from each image, but some images might be too small  
% to find enough. For best performance, you should sample random negative
% examples at multiple scales.

image_files = dir( fullfile( train_path_neg, '*.jpg' ));
num_images = length(image_files);
num_samples = 10000;

% placeholder to be deleted
%features_neg = rand(num_samples, (hog_template_size / hog_cell_size)^2 * 31);

%===================================================================

%        YOUR CODE GOES HERE

% Compute the size of hog feature
feature_size = (hog_template_size/hog_cell_size)^2 * 31;
% Initialize negative feature matrix
features_neg = zeros(num_samples, feature_size);
% Calculate the number of samples to be obtained from each image
sample_per_image = floor(num_samples/num_images);

% Iterate over negative images to sample and compute hog features for each
for i = 1:num_images
    % Read the image
    image = imread(fullfile(image_files(i).folder, image_files(i).name));
    % Convert to grayscale and single
    image_single = im2single(rgb2gray(image));
    % Get the number of rows and columns of the image
    [num_rows, num_cols] = size(image_single);
    % Loop to get samples from the image
    for j = 1:sample_per_image
        % Get random integers as the starting point of sample
        row_start = randi([1,num_rows-hog_template_size]);
        col_start = randi([1,num_cols-hog_template_size]);
        % Calculate the ending point of the sample
        row_end = row_start+hog_template_size-1;
        col_end = col_start+hog_template_size-1;
        % Crop the image to get the sample
        sample = image_single(row_start:row_end, col_start:col_end);
        % Compute hog
        hog = vl_hog(sample, hog_cell_size);
        % Get 1xFEATURE_SIZE feature vector
        feature_vector = reshape(hog, [1, feature_size]);
        % Place the vector into the relevant row in the negative features matrix
        features_neg((i-1) * sample_per_image + j, :) = feature_vector;
    end
end

%==================================================================



end