function [ result, percentage ] = use_nn(image_path)
  % Load image and neural network
  image_original = imread(image_path);
  load('../neural_net/convnetEspecializada.mat');
  % Get only the center of the image
  [y, x, z] = size(image_original);
  image_center = image_original((y/4):3*(y/4), (x/4): 3*(x/4), :);
  % Split the center of the image in N sections and use neural network
  N = 6; M = 6;
  [y2, x2, z2] = size(image_center);
  % highest
  highest_prediction = 0.0;
  highest_class = "";
  for m = 0:(M-1) % iterate through columns (y-axis)
      for n = 0:(N-1) % iterate through rows (x-axis)
          y_start = round(m*(y2/M));
          if y_start == 0
              y_start = 1;
          end
          y_end = round((m+1)*(y2/M));
          x_start = round(n*(x2/N));
          if x_start == 0
              x_start = 1;
          end
          x_end = round((n+1)*(x2/N));
          img_temp = image_center(y_start:y_end, x_start: x_end, :);
          img_resize = imresize(img_temp, [227, 227]);
          class = convnet.classify(img_resize);
          score = max(convnet.predict(img_resize))*100;
          if score > highest_prediction
            highest_prediction = score;
            highest_class = class;
          end
      end
  end
  result = char(highest_class)
  percentage = highest_prediction
end
