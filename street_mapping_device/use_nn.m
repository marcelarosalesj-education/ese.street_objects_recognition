function [ result ] = use_nn(image_path)
  image = imread(image_path);
  image = imresize(image, [227, 227]);
  load('convnetEspecializada.mat');
  result = convnet.predict(image)
end
