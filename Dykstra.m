function [vX] = Dykstra(cProjFun,vY,numIter,stopThr)

numSets = size(cProjFun, 1);
d = sqrt(size(vY,1));
numElements = size(vY, 1);

mZ = zeros(numElements, numSets);
mU = zeros(numElements, numSets);
vU = vY;
vX = vU;

for ii=1:numIter
% while true
%     disp(ii)
    for jj = 1:numSets
       
        
        mU(:, jj) = cProjFun{jj}(vU + mZ(:, jj));
        mZ(:, jj) = vU + mZ(:, jj) - mU(:, jj);
        
        
        vU(:) = mU(:, jj);
    end
    
    % To calculate the difference from the previous iteration.
%     stopCond = (max(abs(vX - vU)) < stopThr);
    stopCond = ((vX - vU)'*(vX - vU) < stopThr);
   
%     disp((vX - vU)'*(vX - vU))
    vX(:) = vU;
    
    
%     pause(0.01);
    
    if(stopCond)
        break;
    end
    
end


end
