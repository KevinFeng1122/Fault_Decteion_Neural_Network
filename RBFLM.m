function [v, w, iter] = RBFLM(x, y, c, gamma, m, eta, epsilon)

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
E = sum(sum((y - yhat).^2)) ;%/ 2;

iter = 1;
NumEtaSplits = 0;
while 1
   
   Eold = E;
   epso = y - yhat;
   
   % ---------- Jacobian Matrix Calculation ----------
   
   Jacob=zeros(1,no*(c+1)+ni*c);
   
   for i = 1 : no
      sumtemp = zeros(1, c+1);
      for k = 1 : M
         sumtemp = sumtemp + epso(i, k) * h(:, k)';
      end
      Jacob((i-1)*(c+1)+1:i*(c+1))=sumtemp'; % w(i, :) = w(i, :) + eta * sumtemp;
   end
   
   for k = 1 : M
      for j = 1 : c
         epsh(j, k) = 2 / (m - 1) * h(j+1, k)^m * epso(:, k)' * w(:, j+1);
      end
   end
   
   for j = 1 : c
      sumtemp = zeros(ni, 1);
      for k = 1 : M
         sumtemp = sumtemp + epsh(j, k) * (x(:, k) - v(:, j)); 
      end
      Jacob(no*(c+1)+(j-1)*ni+1:no*(c+1)+j*ni)=sumtemp; % v(:, j) = v(:, j) + eta * sumtemp;
   end
   
   % ---------- Update Parameters ----------
   
   para_change=(Jacob'*Jacob+eta*eye(no*(c+1)+ni*c))\Jacob';
   
   w_change_vector=para_change(1:no*(c+1));
   
   v_change_vector=para_change(1+no*(c+1):no*(c+1)+ni*c);
   
   w_change_matrix = zeros(no, c+1);
   
   v_change_matrix = zeros(ni, c);
   
   for i1=1:no
       w_change_matrix(i1,:)=w_change_vector((i1-1)*(c+1)+1:i1*(c+1))';
   end
   
   for i2=1:c
       v_change_matrix(:,i2)=v_change_vector((i2-1)*ni+1:i2*ni);
   end
   
   wold = w;
   
   vold = v;
   
   w=w+w_change_matrix;
   
   v=v+v_change_matrix;
   
   
   % ---------- Update Error ----------
   
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
   E = sum(sum((y - yhat).^2));% / 2;

   de = (Eold - E) / Eold;
   disp(['Iteration # ', num2str(iter), ', E = ', num2str(E), ...
         ', de = ', num2str(de)]);

     
   % ---------- Damping Factor Update ----------
   
   if de>0
       eta=eta/10;
   else
       eta=eta*10;
       v = vold;
       w = wold;
       NumEtaSplits = NumEtaSplits + 1;
       if NumEtaSplits > 4
           disp('NumEtaSplits > 4');
          break;
       end       
   end
   
   
   % ---------- Neural Network Training Completion Check ----------
   
   if ((de >= 0) && (de <= epsilon)) || (E <= epsilon)
       disp('Reach Training Threshold');
       break;
   end
   
   
%    if ((de >= 0) && (de <= epsilon)) || (E <= epsilon)
%       break;
%    elseif de < 0
%       v = vold;
%       w = wold;
%       eta = eta / 2;
%       NumEtaSplits = NumEtaSplits + 1;
%       if NumEtaSplits > 4
%          break;
%       end
%    end
   
   iter = iter + 1;
   
end
