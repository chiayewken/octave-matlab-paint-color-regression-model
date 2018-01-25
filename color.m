clear ; close all; clc
data = csvread('colordata.csv');
clc;
fprintf('Color Recipe Machine Learning by Ken v2 \n \n');
X = data(:, 1:16); # 82 * 16
X = [X X .** 2]; # add X square terms # 82 * 32
X = [ones(82, 1) X]; # added bias unit, now 82 * 33
Y = data(:, 17:19); # 82 * 3
theta = zeros(size(X, 2), size(Y, 2)); # 33 * 3
#########################################################
function J = cost(X, Y, theta)
  m = size(X, 1); # 82, number of training examples
  J = 1 / 2 / m * sum((X * theta - Y) .** 2); # average cost across all training examples
end
#########################################################
function [theta] = normal(X, Y)
  theta = pinv(X' * X) * X' * Y;
end
#########################################################
function score = accuracy(X, Y, theta)
  predicted = X * theta; # 82 * 3
  percenterror = (1 - abs(predicted .- Y) ./ Y) * 100;  # 82 * 3
  score = mean(percenterror); # 1 * 3
end
#########################################################
function prediction = check(x, theta)
  prediction = [1 x x .** 2] * theta; # 1 * 3
  plotcolor(prediction(2), prediction(3));
end
#########################################################
function plotcolor(x, y)
color = imread('colorspace.jpg');
image(color);
hold on;
x = (975 - 94) / 0.8 * x + 94;
y = (41 - 917) / 0.9 * y + 917;
plot(x, y, 'linewidth', 5, 'ko', 'markersize', 30);
hold off;
end
#########################################################
# initialcost = cost(X, Y, theta) # check initial cost
theta = normal(X, Y);
# normalcost = cost(X, Y, theta)
score = accuracy(X, Y, theta);
fprintf('Prediction Accuracy \n Y: %f%% \n x: %f%% \n y: %f%% \n \n', score)
fprintf('Use a 1 x 16 vector as input for check(x) function... \n')
fprintf(' e.g. check([0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], theta) \n \n')
