function [ result, percentage ] = use_nn(image_path)
  image = imread(image_path);
  image = imresize(image, [227, 227]);
  load('../neural_net/convnetEspecializada.mat');
  result = char(convnet.classify(image));
  percentage = convnet.predict(image);
end
