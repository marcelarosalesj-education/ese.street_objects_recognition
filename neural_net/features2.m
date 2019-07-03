net = alexnet; % Returns a AlexNet net pretrained

layers = net.Layers;
layers; 

rootFolder = 'Train'; % Train classes to train the net

categories = {'0 Max Speed 20', '1 Max Speed 30', '2 Max Speed 50', '3 Max Speed 60', '4 Max Speed 70', ...
    '5 Max Speed 80', '6 End of Speed Lim', '7 Max Speed 100', '8 Max Speed 120', '9 No overtaking', ...
    '10 No overtaking for trailers', '11 Intersection with priority', '12 Priority road starts', ...
    '13 Give away', '14 Stop', '15 No vehicles', '16 No trailers', '17 No entry for vehicular traffic', ...
    '18 Danger', '19 Left bend', '20 Right bend', '21 Double bend', '22 Uneven road', '23 Road slippery', ...
    '24 Road narrows', '25 Road works', '26 Light signals', '27 Pedestrian crossing', '28 Children', ...
    '29 Cyclists crossing', '30 Danger of snow or ice', '31 Wild animals crossing', '32 End of prohibitions', ...
    '33 Turn right ahead', '34 Turn left ahead', '35 Ahead only', '36 Ahead or right only', ...
    '37 Ahead or left only', '38 Pass on the right', '39 Pass on the left', '40 Right of way traffic in the circle', ...
    '41 End of prohibition to overtake', '42 End of prohibition to overtake for power trailers'};

% Manage a collection of image files
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');

% Split image files into two sets, in this case only one
[trainingSet, ~] = splitEachLabel(imds, 1); % 1 absolute number of files for the label
imds.ReadFcn = @readFunctionTrain; % Resize images according to net input size

layers = layers(1:end-3);
layers(end+1) = fullyConnectedLayer(64, 'Name', 'special_2');
layers(end+1) = reluLayer;
layers(end+1) = fullyConnectedLayer(43, 'Name', 'fc8_2 '); %Number of clases
layers(end+1) = softmaxLayer;
layers(end+1) = classificationLayer();

layers(end-2).WeightLearnRateFactor = 10;
layers(end-2).WeightL2Factor = 1;
layers(end-2).BiasLearnRateFactor = 20;
layers(end-2).BiasL2Factor = 0;

% Training Options
% MiniBatchSize - how many pictures can be used for each iteration
% MaxEpochs - an epoch is a train cycle
% InitialLearnRate - learning delay
miniBatchSize = 128;
%valFrequency = 450;%58649/miniBatchSize;%.Files
opts = trainingOptions('sgdm', ...
    'LearnRateSchedule', 'none',...
    'InitialLearnRate', .0001,... 
    'MaxEpochs', 20, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress');%AnMer
%'ValidationData',augvalidationSet, ...
%'ValidationFrequency',valFrequency, ...
%'Shuffle','every-epoch', ...
%'Verbose',false, ...
	
convnet = trainNetwork(imds, layers, opts); %netTransfer

rootFolder = 'Test';
testSet = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
testSet.ReadFcn = @readFunctionTrain;  
%augvalidationSet = augmentedImageDatastore([227 227],testSet);

[labels,err_test] = classify(convnet, testSet, 'MiniBatchSize', 64);

confMat = confusionmat(testSet.Labels, labels);
confMat = confMat./sum(confMat,2);
mean(diag(confMat))

%[YPred,probs] = classify(convnet,testSet);
%accuracy = mean(YPred == validationSet.Labels);

%numTrainImages = numel(trainingSet.Labels);
%idx = randperm(numTrainImages,16);
%figure
%for i = 1:16
 %   subplot(4,4,i)
  %  I = readimage(trainingSet,idx(i));
   % imshow(I)
%end

save convnetEspecializada convnet

function I = readFunctionTrain(filename)
% Resize the images to the size required by the network.
I = imread(filename);
I = imresize(I, [227 227]);
end