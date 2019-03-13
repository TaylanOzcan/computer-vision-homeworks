function [bboxes, confidences, image_ids] = ....
    run_detector(test_data_path, svmClassifier, hog_template_size,hog_cell_size)

% This function returns detections on all of the images in 'test_data_path'.

% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.

% INPUT:
% . test_data_path: a string which is path to the directory that contains
%       images which may or may not have faces in them. This function
%       should work for the MIT+CMU test set but also for any other images
%       (e.g. class photos)
% . svmClassifier: A struct with two fields, 'Weights' and 'Bias' that are
%       the parameters of a linear classifier
% . hog_template_size: size of hog template.(default 36)
% . hog_cell_size: the number of pixels in each HoG cell (default 6). Image size
%       should be evenly divisible by hog_cell_size.

% OUTPUT:
% . bboxes: a Nx4 matrix. N is the number of detections. bboxes(i,:) is
%       [x_min, y_min, x_max, y_max] for detection i.
%       Remember 'y' is dimension 1 in Matlab!
% . confidences: a Nx1 matrix. confidences(i) is the real valued confidence of
%       detection i.
% . image_ids: a Nx1 cell array. image_ids{i} is the image file name
%       for detection i. (not the full path, just 'albert.jpg')


test_scenes = dir( fullfile( test_data_path, '*.jpg' ));
num_images = length(test_scenes);

%initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);

for i = 1:num_images
    
    image_name = test_scenes(i).name;
    fprintf('Detecting faces in %s\n', image_name);
    img = imread( fullfile( test_data_path, test_scenes(i).name ));
    
    % The PlaceHolder() function is here to help you understand how the
    % outputs (bboxes,confidences,image_ids) should be structured. This
    % function generates random bounding boxes for a test image. It will
    % even do non-maximum suppression on the random bounding boxes to give
    % you an example of how to call the function. This part of the code needs
    % to be commented out when you write your own function.
    
    %[cur_confidences,cur_bboxes] =...
    %    PlaceHolder(img, svmClassifier, hog_template_size,hog_cell_size);
    
    
    % You will be coding the Detector() function. It has a similar
    % function definition as that of PlaceHolder().
    
    %  Complete the function Detector() and uncomment the line below
    [cur_confidences,cur_bboxes] =...
        Detector(img, svmClassifier, hog_template_size,hog_cell_size);
    
    cur_image_ids = cell(0,1);
    cur_image_ids(1:size(cur_bboxes,1)) = {test_scenes(i).name};
    
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end
end



function [cur_confidences,cur_bboxes] = ...
    PlaceHolder( img,svmClassifier, hog_template_size,hog_cell_size)


% Creating 15 random detections per image
cur_x_min = rand(15,1) * size(img,2);
cur_y_min = rand(15,1) * size(img,1);
cur_bboxes = [cur_x_min, cur_y_min, cur_x_min + rand(15,1) * 50, cur_y_min + rand(15,1) * 50];
cur_confidences = rand(15,1) * 4 - 2; %confidences in the range [-2 2]


%non_max_supr_bbox can actually get somewhat slow with thousands of
%initial detections. You could pre-filter the detections by confidence,
%e.g. a detection with confidence -1.1 will probably never be
%meaningful. You probably _don't_ want to threshold at 0.0, though. You
%can get higher recall with a lower threshold. You don't need to modify
%anything in non_max_supr_bbox, but you can.

[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences,size(img));
end


function [bboxes, confidences] = ...
    nonMaximum_Suppression(bboxes, confidences,img_size)

[is_maximum] = non_max_supr_bbox(bboxes, confidences, img_size);
confidences = confidences(is_maximum,:);
bboxes      = bboxes(is_maximum,:);
end


function [cur_confidences,cur_bboxes] = ...
    Detector(img, svmClassifier, hog_template_size,hog_cell_size)


% Your code should first convert each test image to grayscale if
% it is a coloured image.

% Then the HoG features need to be determined for the entire image with
% a single call to vl_hog for each scale.

% Then step over the HoG cells, taking groups of cells that are the
% same size as your learned template, and classify them. Do this for all sclaes.

% If the classification is above some confidence, keep the detection
% and then pass all the detections for an image to non-maximum
% suppression.

% For your initial debugging, you can operate only at a
% single scale. A word of advise. Don't save all the HoGs. Save
% the bounding boxes and confidences of only detected faces.
% This will speed up your detections.



cur_bboxes = zeros(0,4);
cur_confidences = zeros(0,1);
%==============================================================

%       YOUR CODE HERE

% If image has 3rd dimension (colored)
if size(img,3)>1
    % Convert to grayscale
    img = rgb2gray(img);
end
% Convert to single
img = im2single(img);

% Compute the size of hog feature
feature_size = 31*(hog_template_size/hog_cell_size)^2;
% Calculate the number of cells in a template
num_cells = hog_template_size/hog_cell_size;

% Get the weight and bias terms from the classifier
w = svmClassifier.weights;
b = svmClassifier.bias;

% Set a threshold value to compare scores to decide whether face or not
threshold = 0.8;
% Set the starting scale
scale = 1.25;

% Loop over different scales until scale>0.25
while scale > 0.25
    % Scale the image
    img_scaled = imresize(img, scale);
    % Compute hog for the scaled image
    hog = vl_hog(img_scaled, hog_cell_size);
    % Get the size of the hog matrix
    [height, width, feature_length] = size(hog);
    
    % Loop over each hog cell (to simulate sliding window)
    for i = 1:height-num_cells
        for j = 1:width-num_cells
            % Get the current hog window
            cur_window = hog(i:i+num_cells-1,j:j+num_cells-1,:);
            % Get 1xFEATURE_SIZE feature vector for current window
            cur_feature = reshape(cur_window, [feature_size, 1]);
            % Calculate the score value for current feature
            score = w' * cur_feature + b;
            % If score > threshold, then the window contains a face
            if score > threshold
                % Compute size of the bounding box for the current scale
                bbox_size = hog_template_size/scale;
                % Compute the starting points of the current bounding box
                cur_x_min = (hog_cell_size*(j-1)+1)/scale;
                cur_y_min = (hog_cell_size*(i-1)+1)/scale;
                % Compute the ending points of the current bounding box
                cur_x_max = cur_x_min+bbox_size - 1;
                cur_y_max = cur_y_min+bbox_size - 1;
                % Set the boundaries of the current bounding box
                cur_bbox = [cur_x_min, cur_y_min, cur_x_max , cur_y_max];
                % Add current bounding box into bounding boxes vector
                cur_bboxes = [cur_bboxes; cur_bbox];
                % Add current confidence value into confidences vector
                cur_confidences = [cur_confidences; score];
            end
        end
    end
    % Decrease the scale by multiplying with 0.9 at each step 
    scale = scale * 0.9;
end

% Pass the detections to non-maximum suppression
[cur_bboxes, cur_confidences] = ...
    nonMaximum_Suppression(cur_bboxes, cur_confidences, size(img));

%==============================================================


end