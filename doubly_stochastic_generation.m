%Randomly generate a doubly sotchastic matrix
function [P] = doubly_stochastic_generation(agent_num)
    for i=1:2
        x=rand(agent_num,1);
        x=x/sum(x);
        x = x';
        
        A{i} = toeplitz([x(1) fliplr(x(2:end))], x);
        A{i}= A{i}(randperm(agent_num),randperm(agent_num));
    end
    
    w = rand;
    
    P = w*A{1} + (1-w)*A{2};
end