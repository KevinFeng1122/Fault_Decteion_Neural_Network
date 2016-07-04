function [Pct, Err, yhat] = RBFTest(x, y, v, w, gamma, m)

% function [Pct, Err] = RBFTest(x, y, v, w, gamma, m)
% Radial basis function test using linear generator functions.
%
% INPUTS
% x = test inputs, an ni x M matrix, where 
%     ni is the dimension of each input, and
%     M is the total number of training vectors.
% y = correct test outputs, an no x M matrix, where
%     no is the dimension of each output, and
%     M is the total number of training vectors.
% v = prototypes at middle layer, an ni x c matrix.
% w = weight matrix between middle layer and output layer,
%     an no x (c+1) matrix.
% gamma = generator function parameter.
% m = generator function parameter (integer greater than one).
%
% OUTPUTS
% Pct = Percent of test cases correctly classified.
% Err = RMS error of test outputs.

M = size(x, 2);
if M ~= size(y, 2)
   disp('Inconsistent matrix sizes');
   return;
end
ni = size(x, 1);
if ni ~= size(v, 1)
   disp('Inconsistent matrix sizes');
   return;
end
no = size(y, 1);
if no ~= size(w, 1)
   disp('Inconsistent matrix sizes');
   return;
end
c = size(v, 2);
if (c + 1) ~= size(w, 2)
   disp('Inconsistent matrix sizes');
   return;
end
   
gamma2 = gamma * gamma;
h = ones(c+1, M);

for k = 1 : M
   for j = 1 : c
      diff = norm(x(:, k) - v(:, j))^2;
      if (diff + gamma2) < eps
         h(j+1, k) = 0;
      else
         h(j+1, k) = (diff + gamma2) ^ (1 / (1 - m));
      end
   end
end
   
for i = 1 : no
   for k = 1 : M
      yhat(i, k) = w(i, :) * h(:, k);
   end
end

Err = 0;
nCorrect = 0;
for k = 1 : M
   ymax = -realmax;
   yhatmax = -realmax;
   for i = 1 : no
      Err = Err + (y(i, k) - yhat(i, k))^2;
      if y(i, k) > ymax
         ymax = y(i, k);
         Class = i;
      end
      if yhat(i, k) > yhatmax
         yhatmax = yhat(i, k);
         ClassHat = i;
      end
   end
   if Class == ClassHat
      nCorrect = nCorrect + 1;
   end
end
Err = sqrt(Err / M);
Pct = 100 * nCorrect / M;
