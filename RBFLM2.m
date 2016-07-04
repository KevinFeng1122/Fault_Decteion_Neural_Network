function [v, w, iter] = RBFLM2(x, y, c, gamma, p, eta, epsilon)

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
% p = generator function parameter (integer greater than one).
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

w = rand(no, c+1);%zeros(no, c+1);
v = zeros(ni, c);
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
         h(j+1, k) = (diff + gamma2) ^ (1 / (1 - p));
      end
   end
end

yhat = w * h;
E = sum(sum((y - yhat).^2)) ;%/ 2;

iter = 1;
NumEtaSplits = 0;
while 1
   
   epso = y - yhat;
   
   % ---------- Jacobian Matrix Calculation ----------
      
   Jacob_weight=zeros(no*M,no*(c+1));
   
   Jacob_proto=zeros(no*M,ni*c);
   
   
   for i=1:no
       for j=1:M
           for a=1:no
               for b=0:c
                   if a==i
                       Jacob_weight((j-1)*no+i,(a-1)*(c+1)+(b+1))=-h(b+1,j);
                   end                   
               end
           end
       end
   end
   
   
   for i=1:no
       for j=1:M
           for a=1:ni
               for b=1:c
                   Jacob_proto((j-1)*no+i,(a-1)*ni+b)=...
                       2*w(i,b)/(1-p)*(h(b+1,j)^p)*(x(a,j)-v(a,b));
               end
           end
       end
   end
   
   Jacob=[Jacob_weight,Jacob_proto];
   
   
   % ---------- Parameter Update Calculation ----------
   
   d=zeros(no*M,1);
   for i=1:M
       d(no*(i-1)+1:no*i,1)=epso(:,i);
   end
   
   para_change=(Jacob'*Jacob+eta*eye(no*(c+1)+c*ni))\Jacob'*d;
   
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
   
   w_old = w;
   
   v_old = v;
   
   w=w+w_change_matrix;
   
   v=v+v_change_matrix;
  
   
   % ---------- Update Error ----------
   
   h_old=h;
   
   for j = 1 : c
      for k = 1 : M
         diff = norm(x(:, k) - v(:, j))^2;
         if (diff + gamma2) < eps
            h(j+1, k) = 0;
         else
            h(j+1, k) = (diff + gamma2) ^ (1 / (1 - p));
         end
      end
   end
   
   yhat = w * h;
   E_old = E;
   E = sum(sum((y - yhat).^2));% / 2;

   de = E_old - E;
   disp(['Iteration # ', num2str(iter), ', E = ', num2str(E), ...
         ', de = ', num2str(de)]);

     
   % ---------- Damping Factor Update ----------
   
   if de>0
       eta=eta/10;
   else
       eta=eta*10;
       v=v_old;
       w=w_old;
       h=h_old;
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
   
   iter = iter + 1;
   
end
