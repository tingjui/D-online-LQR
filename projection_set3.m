function [Sigma] = projection_set3(Sigma,A,B,W)

[d,k] = size(B);

%Matrices defining the equality constraints
S = [A B]; 
E = [eye(d) zeros([d k])];

C = zeros([d^2 (d+k)^2]);

row_index = 0;
for i=1:d
    for j=1:d
        row_index = row_index + 1;
        C(row_index,:) = reshape((S(j,:)'*S(i,:)-E(j,:)'*E(i,:))',[],1)';
    end
end

new_C = zeros([d^2 (d+k+1)*(d+k)/2]);

counter = 0;
for i=1:(d+k)
   for j=i:(d+k)
       counter = counter + 1;
       if i==j
           new_C(:,counter) = C(:,(d+k)*(i-1)+j);
       else
           new_C(:,counter) = C(:,(d+k)*(i-1)+j) + C(:,(d+k)*(j-1)+i);
       end
   end    
end
C = new_C;

% lambda = (C*C')\(-W-C*Sigma);
lambda = pinv(C*C')*(-W-C*Sigma);
Sigma = Sigma + C'*lambda;
% disp(eig(inv(C*C')))
% disp(C*Sigma + W)
end