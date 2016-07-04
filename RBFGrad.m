function [v, w, iter, Error_all] = RBFGrad(x, y, c, gamma, m, epsilon, eta)

% function [v, w, iter] = RBFGrad(x, y, c, gamma, m, eta, epsilon)
% Radial basis function training using linear generator functions
% and gradient descent.
%
% INPUTS
% x = training inputs, an ni x M matrix, where 
%     ni is the dimension of each input, and
%     M is the total number of training vectors.
% y = training outputs, an no x M matrix, where
%     no is the dimension of each output, and
%     M is the total number of training vectors.
% c = # of radial basis function centers.
% gamma = generator function parameter (typically between 0 and 1).
% m = generator function parameter (integer greater than one).
% eta = gradient descent step size.
% epsilon = delta-error threshold at which to stop training.
%
% OUTPUTS
% v = prototypes at middle layer, an ni x c matrix.
% w = weight matrix between middle layer and output layer, an no x (c+1) matrix.
% iter = # of iterations it took to converge.

M = size(x, 2);
if M ~= size(y, 2)
   disp('Inconsistent matrix sizes');
   return;
end
ni = size(x, 1);
no = size(y, 1);

gamma2 = gamma * gamma;

w = zeros(no, c+1);
v = zeros(ni, c);
epsh = zeros(c+1, M);

h = ones(c+1, M);

for i = 0 : c-1
   v(:, i+1) = x(:, round(M*i/c) + 1);
end

for j = 1 : c
   for k = 1 : M
      diff = norm(x(:, k) - v(:, j))^2;
      if (diff + gamma2) < eps
         h(j+1, k) = 0;
      else
         h(j+1, k) = (diff + gamma2) ^ (1 / (1 - m));
      end
   end
end

yhat = w * h;
E = sum(sum((y - yhat).^2)) / 2;

iter = 1;
NumEtaSplits = 0;

Error_all=[];

while 1
   
   Eold = E;
   epso = y - yhat;
   
   wold = w;
   for i = 1 : no
      sumtemp = zeros(1, c+1);
      for k = 1 : M
         sumtemp = sumtemp + epso(i, k) * h(:, k)';
      end
      w(i, :) = w(i, :) + eta * sumtemp;
   end
   
   for k = 1 : M
      for j = 1 : c
         epsh(j, k) = 2 / (m - 1) * h(j+1, k)^m * epso(:, k)' * w(:, j+1);
      end
   end
   
   vold = v;
   for j = 1 : c
      sumtemp = zeros(ni, 1);
      for k = 1 : M
         sumtemp = sumtemp + epsh(j, k) * (x(:, k) - v(:, j));
      end
      v(:, j) = v(:, j) + eta * sumtemp;
   end
   
   for j = 1 : c
      for k = 1 : M
         diff = norm(x(:, k) - v(:, j))^2;
         if (diff + gamma2) < eps
            h(j+1, k) = 0;
         else
            h(j+1, k) = (diff + gamma2) ^ (1 / (1 - m));
         end
      end
   end
   
   yhat = w * h;
   E = sum(sum((y - yhat).^2)) / 2;
   
   Error_all=[Error_all;E];

   de = (Eold - E) / Eold;
   disp(['Iteration # ', num2str(iter), ', E = ', num2str(E), ...
         ', de = ', num2str(de)]);

   if ((de >= 0) & (de <= epsilon)) | (E <= epsilon)
      break;
   elseif de < 0
      v = vold;
      w = wold;
      eta = eta / 2;
      NumEtaSplits = NumEtaSplits + 1;
      if NumEtaSplits > 4
         break;
      end
   end
   
   iter = iter + 1;
   
end
