% Implementation of EXTRA

d = 3;  %Dimension of the state 
k = 3;  %Dimension of the control vector
agent_num = 5; %Number of agents




% W = reshape(eye(d)*(1),[],1);
lamb_squared = d;
sigma_squared = 1;
Coeffi = eye(d+k);
Coeffi = Coeffi(tril(true(d+k)));
gamma = 0.4;
kappa = 1.5;
K_s = eye(d)*kappa;
A = eye(d)*(1-2*gamma);
B = eye(d)*(gamma/kappa);
v_squared = (1-2*gamma)^2 + (gamma/kappa)^2;
kappa_zero = 0.5*kappa;
gamma_zero = 1.5*gamma;



%Generation of Doubly Stochastic Matrix
self_weight = 0.6;
weight = [self_weight (1-self_weight)/2 0 0 (1-self_weight)/2];
P = toeplitz([weight(1) fliplr(weight(2:end))], weight);
s = svds(P);
beta = s(2);

new_P = (eye(agent_num) + P);
P_tilde = new_P./2;

T_zero = 1000000;
[eigenvector, eigenvalue] = eig(P_tilde);
lambda = sigma_squared/v_squared;
lipschitz_contant = 2*((4*sqrt(sigma_squared)*kappa_zero/gamma_zero)^2)*(d+k*v_squared*kappa_zero^2)*log(4*T_zero)*T_zero + 2*lambda/agent_num;
step_size = (2*lambda/agent_num*min(diag(eigenvalue)))/lipschitz_contant^2;
step_size = step_size*1e9;

%%

system_estimate = zeros(d,d+k,agent_num);
system_estimate_pre = zeros(d, d+k, agent_num);
system_estimate_ppre = rand(d, d+k, agent_num);

learner_state_collect = zeros(d, agent_num, T_zero);
learner_control_collect = zeros(k, agent_num, T_zero);

iteration_num = 1000;
diff_norm = zeros(agent_num, iteration_num);

% Collect the data for each agent
learner_state = zeros(d, agent_num);
learner_control = zeros(k, agent_num);
K_g = eye(d)*(kappa_zero);
for i=1:T_zero
    if i==1
        leanrner_state = mvnrnd(zeros(d,1),eye(d)*10, agent_num)';
    else
        noise = mvnrnd(zeros(d,1),eye(d),agent_num)';
        learner_state = A*learner_state + B*learner_control + noise;   
    end

    learner_state_collect(:,:,i) = learner_state;
    for m=1:agent_num
        learner_control(:,m) = mvnrnd(K_g*learner_state(:,m),2*sigma_squared*(kappa_zero^2)*eye(d),1)';
    end
    learner_control_collect(:,:,i) = learner_control;

end

ZZ_prime_s = zeros(agent_num, d+k, d+k);
XZ_prime_s = zeros(agent_num, d, d+k);

for i=1:agent_num
   Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
   Z = squeeze(Z);
   ZZ_prime_s(i,:,:) = Z*Z';
   XZ_prime_s(i,:,:) = squeeze(learner_state_collect(:,i,2:T_zero))*Z';
end 
    
% Compute the exact minimizer
V_zero = zeros(d+k,d+k);
product_term = zeros(d,d+k);
for i=1:agent_num
%     Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
%     Z = squeeze(Z);
%     V_zero = V_zero + (Z*Z'); 
%     product_term = product_term + squeeze(learner_state_collect(:,i,2:T_zero))*Z';
    V_zero = V_zero + squeeze(ZZ_prime_s(i,:,:));
    product_term = product_term + squeeze(XZ_prime_s(i,:,:));
    
end
V_zero  = V_zero + eye(d+k)*sigma_squared/v_squared;
system_minimizer = product_term/V_zero;


% EXTRA
for i=1:agent_num
    iterate_weight_avg = zeros(d, d+k);
    for j=1:agent_num
       iterate_weight_avg = iterate_weight_avg + P(j,i)*system_estimate_ppre(:,:,j);
    end
%     Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
%     Z = squeeze(Z);
%     gradient = 2*system_estimate_ppre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
%         + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);
    
    gradient = 2*system_estimate_ppre(:,:,i)*squeeze(ZZ_prime_s(i,:,:)) - 2*squeeze(XZ_prime_s(i,:,:))...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);
    
    system_estimate_pre(:,:,i) = iterate_weight_avg - step_size*gradient;
end

for iter=1:iteration_num
   for i=1:agent_num
      iterate_weight_avg1 = zeros(d, d+k);
      iterate_weight_avg2 = zeros(d, d+k);
      
      for j=1:agent_num
         iterate_weight_avg1 = iterate_weight_avg1 + new_P(j,i)*system_estimate_pre(:,:,j);   
         iterate_weight_avg2 = iterate_weight_avg2 + P_tilde(j,i)*system_estimate_ppre(:,:,j);
      end    
%       Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
%       Z = squeeze(Z);
%       
%       gradient1 = 2*system_estimate_pre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
%         + 2*(sigma_squared/v_squared)/agent_num*system_estimate_pre(:,:,i);
%     
%       gradient2 = 2*system_estimate_ppre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
%         + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);
    
      gradient1 = 2*system_estimate_pre(:,:,i)*squeeze(ZZ_prime_s(i,:,:)) - 2*squeeze(XZ_prime_s(i,:,:))...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_pre(:,:,i);
    
      gradient2 = 2*system_estimate_ppre(:,:,i)*squeeze(ZZ_prime_s(i,:,:)) - 2*squeeze(XZ_prime_s(i,:,:))...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);  
    
      system_estimate(:,:,i) = iterate_weight_avg1 - iterate_weight_avg2 - step_size*(gradient1 - gradient2);
 
   end
   
   for i=1:agent_num
       diff_norm(i,iter) =  norm(system_minimizer - system_estimate(:,:,i), 'fro');
   end
   
   system_estimate_ppre = system_estimate_pre;
   system_estimate_pre = system_estimate;
end    

%%

figure;

for i=1:agent_num
    plot(diff_norm(i,1:iteration_num));
    hold on
end
hold off

% sd = sum(diff_norm.^2,1);
% figure;
% plot(sd(1:100));

%%
loss_minimizer = 0;
loss_convergence = 0;

for i=1:agent_num
    Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
    Z = squeeze(Z);
    for j=1:T_zero-1
       loss_minimizer = loss_minimizer + norm(system_minimizer*Z(:,j) - learner_state_collect(:,i,j+1))^2;
       loss_convergence = loss_convergence + norm(system_estimate(:,:,1)*Z(:,j) - learner_state_collect(:,i,j+1))^2;
    end
end

loss_minimizer = loss_minimizer + (sigma_squared/v_squared)*norm(system_minimizer,'fro'); 
loss_convergence = loss_convergence + (sigma_squared/v_squared)*norm(system_estimate(:,:,1),'fro'); 

%%
% loss1 = 0;
% loss2 = 0;
% 
% Z = [learner_state_collect(:,1,1:T_zero-1);learner_control_collect(:,1,1:T_zero-1)];
% Z = squeeze(Z);
% gradient = 2*system_estimate(:,:,1)*(Z*Z') - 2*squeeze(learner_state_collect(:,1,2:T_zero))*Z'...
%     + 2*(sigma_squared/v_squared)/agent_num*system_estimate(:,:,1);
% 
% updated = system_estimate(:,:,1) - step_size*gradient;
% 
% for j=1:T_zero-1
%    loss1 = loss1 + norm(system_estimate(:,:,1)*Z(:,j) - learner_state_collect(:,1,j+1))^2;
%    loss2 = loss2 + norm(updated*Z(:,j) - learner_state_collect(:,1,j+1))^2;
% end
% 
% loss1 = loss1 + (sigma_squared/v_squared/agent_num)*norm(system_estimate(:,:,1),'fro'); 
% loss2 = loss2 + (sigma_squared/v_squared/agent_num)*norm(updated,'fro'); 









