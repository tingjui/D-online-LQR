%Projection to a half space
function [Sigma] = projection_set2(Sigma,Coeffi,nu)
%I = reshape(eye(size(Sigma)),[],1);

if Coeffi'*Sigma > nu 
    Sigma = Sigma - (Coeffi'*Sigma - nu)/(Coeffi'*Coeffi)*Coeffi;
end

end