function [v, w, iter, Error_all] = RBFKalman(x, y, c, gamma, m, epsilon, P0, Q0, R0)

% Function [v, w, iter] = RBFKalman(x, y, c, gamma, m, epsilon, P0, Q0, R0)
% Radial basis function training using linear generator functions
% and Kalman filtering.
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
% epsilon = delta-error threshold at which to stop training.
% P0 = initial setting of estimate covariance matrix (about 40).
% Q0 = initial setting of state covariance matrix (about 40).
% R0 = initial setting of measurement covariance matrix (about 40).
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

h = ones(c+1, M);
n = no * (c + 1) + ni * c;
P = P0 * eye(n);
Q = Q0 * eye(n);
R = R0 * eye(no * M);

for i = 0 : c-1
   v(:, i+1) = x(:, round(M*i/c) + 1);
end

% Put the RBF parameters in the Kalman filter state vector.
xhat = [];
for i = 1 : no
   xhat = [xhat ; w(i, :)'];
end
for i = 1 : c
   xhat = [xhat ; v(:, i)];
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
disp(['Initial E = ', num2str(E)]);

iter = 1;
NumPDoubles = 0;

Error_all=[];

while 1

   Eold = E;
   vold = v;
   wold = w;
   
   % Compute the partial derivative of the error with respect to
   % the components of the prototypes in the v matrix.
   vpartial = [];
   for i = 1 : no
      for k = 1 : M
         vpartialcol = [];
         for j = 1 : c
            vpartialcol = [vpartialcol ; -w(i, j) / (1 - m) * ...
                           h(j+1, k)^m * 2 * (x(:, k) - v(:, j))];
         end
         vpartial = [vpartial vpartialcol];
      end
   end

   % Compute the partial derivative of the error with respect to
   % the components of the weight matrix w.
   wpartial = zeros(no*(c+1), no*M);
   for i = 1 : no
      wpartial((i-1)*(c+1)+1 : i*(c+1), (i-1)*M+1 : i*M) = h;
   end

   % Compute the RBF error vector.
   errorvector = [];
   for i = 1 : no
      for k = 1 : M
         errorvector = [errorvector ; y(i, k) - yhat(i, k)];
      end
   end
   
   % Compute the Kalman gain.
   H = [wpartial ; vpartial];
   K = P * H * inv(R + H' * P * H);
   
   % Update the state estimate.
   xhat = xhat + K * errorvector;
   
   % Update the covariance matrix.
   P = P - K * H' * P + Q;
   
   % Extract w and v from xhat.
   for i = 1 : no
      StartIndex = (i - 1) * (c + 1) + 1;
      w(i, :) = xhat(StartIndex : StartIndex + c)';
   end
   for i = 1 : c
      StartIndex = no * (c + 1) + (i - 1) * ni + 1;
      v(:, i) = xhat(StartIndex : StartIndex + ni - 1);
   end
   
   % Based on the new w and v matrices, compute the output 
   % of the RBF network.
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
      P = 2 * P;
      NumPDoubles = NumPDoubles + 1;
      if NumPDoubles > 1
         break;
      end
   end
  
   iter = iter + 1;
   
end