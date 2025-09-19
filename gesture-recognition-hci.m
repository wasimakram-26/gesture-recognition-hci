l%% GESTURE RECOGNITION USING MATLAB

clc; clear; close all;

%% Step 1: Load gesture dataset
gestureDB = imageDatastore('gesture_dataset', ...
    'IncludeSubfolders', true, ...
    'LabelSource','foldernames');

% Split into training & testing
[trainImgs, testImgs] = splitEachLabel(gestureDB,0.8,'randomized');

%% Step 2: Extract HOG features for all images
trainFeatures = [];
trainLabels = trainImgs.Labels;
for i = 1:numel(trainImgs.Files)
    img = readimage(trainImgs,i);
    img = imresize(rgb2gray(img),[64 64]);
    feature = extractHOGFeatures(img);
    trainFeatures = [trainFeatures; feature];
end

% Train SVM classifier
classifier = fitcecoc(trainFeatures, trainLabels);

%% Step 3: Test accuracy
testFeatures = [];
testLabels = testImgs.Labels;
for i = 1:numel(testImgs.Files)
    img = readimage(testImgs,i);
    img = imresize(rgb2gray(img),[64 64]);
    feature = extractHOGFeatures(img);
    testFeatures = [testFeatures; feature];
end

predictedLabels = predict(classifier, testFeatures);
accuracy = mean(predictedLabels == testLabels);
fprintf('Test Accuracy: %.2f%%\n', accuracy*100);

%% Step 4: Real-time gesture recognition via webcam
cam = webcam;
disp('Press Ctrl+C to stop webcam');

while true
    img = snapshot(cam);
    gray = rgb2gray(img);
    
    % Simple segmentation using thresholding
    bw = imbinarize(gray,'adaptive','ForegroundPolarity','dark');
    bw = bwareaopen(bw,500);
    
    % Find largest region (assumed hand)
    stats = regionprops(bw,'BoundingBox','Area');
    if ~isempty(stats)
        [~,idx] = max([stats.Area]);
        hand = imcrop(gray, stats(idx).BoundingBox);
        hand = imresize(hand,[64 64]);
        features = extractHOGFeatures(hand);
        gesture = predict(classifier, features);
        
        % Display result
        img = insertObjectAnnotation(img,'rectangle',stats(idx).BoundingBox,gesture,'Color','green','FontSize',14);
    end
    
    imshow(img);
    drawnow;
end