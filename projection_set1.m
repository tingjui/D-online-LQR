%Project a symmetric matrix to the cone of PSD matrices
function [Sigma] = projection_set1(Sigma, dim)
y = zeros(dim);
lowerTriangleIndices = tril(true(dim));
y(lowerTriangleIndices) = Sigma;
Sigma = y + y.' - diag(diag(y));

% disp(Sigma)

[V,D] = eig(Sigma);
D(D<0)=0;
Sigma = V*D*V';

Sigma  = Sigma(lowerTriangleIndices);


end