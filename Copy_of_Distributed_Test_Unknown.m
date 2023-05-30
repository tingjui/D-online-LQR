d = 3;  %Dimension of the state 
k = 3;  %Dimension of the control vector
agent_num = 20; %Number of agents

numSets = 3;

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

%Generation of Doubly Stochastic Matrix
self_weight = 0.6;
% P = doubly_stochastic_generation(agent_num);
% P = ones(agent_num,agent_num)/agent_num;
% P = eye(agent_num)*(self_weight - (1-self_weight)/(agent_num-1)) + ...
%     ones(agent_num,agent_num)*(1-self_weight)/(agent_num-1);
% beta = 1-agent_num*min(min(P));

weight = [self_weight (1-self_weight)/2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 (1-self_weight)/2];
P = toeplitz([weight(1) fliplr(weight(2:end))], weight);
s = svds(P);
beta = s(2);


% %Define the dynamic matrices
% V = eye(d)*tril(randi([1,10],d,d));
% B = V(:,1:k);
% A = B*randi([1,10],k,d);
% % A = randi([1,10],d,d);
% [U,R]=gschmidt(V);
% H = U*diag([rand(1,k)*0.8 zeros(1,d-k)])*U';
% K = (B'*B)\B'*(H-A);
% M = A +B*K;
% 
% %Generate a strongly stable controller with respect to [A B]
% % gamma = 1-max(svds(M));
% % P = eye(d)/(eye(d)-M'*M/((1-gamma)^2));
% % e = eig(P);
% % K_s = K;
% % kappa = cond(diag(sqrt(e)));
% % kappa = max([svds(K_s);kappa]);
% gamma = 0.6;
% L = diag([rand(1,k)*(1-gamma) zeros(1,d-k)]);
% diag_vec = rand(d,1)+5;
% H = diag(diag_vec);
% H_inv = diag(1./diag_vec);
% K_s = (B'*B)\B'*(U*H*L*H_inv*U' - A);
% kappa = max([svds(K_s);cond(H)]);

%Define dependent hyperparameters
C = 100*d;

nu = 2*kappa^4*lamb_squared/gamma; 

rho = 4*agent_num*C^2*(3+(4*sqrt(agent_num))/(1-beta)) + (1+nu/sigma_squared)*16*sqrt(2)*agent_num^(3/2)...
    *C^(2)*nu^2/(1-beta)/sigma_squared^2;
T = ((4*sqrt(2)*sqrt(agent_num))*nu*C/(sigma_squared^2*(1-beta)*rho^(1/4)))^2;

%%


M = A+B*K_s;


cov_x = inv(eye(d)-M*M');
% cov_x = (eye(d)-M*M')\eye(d);
cov = [cov_x cov_x*K_s';K_s*cov_x K_s*cov_x*K_s'];
%cov_x = inv(eye(d)-(A+B*K))*eye(d)*inv(eye(d)-(A+B*K))';

%Dykstra's Algorithm
W = reshape(eye(d)*(1),[],1);
cProjFun = cell(numSets, 1);
cProjFun{1} = @(Sigma) projection_set1(Sigma,d+k);
cProjFun{2} = @(Sigma) projection_set2(Sigma,Coeffi,nu);
cProjFun{3} = @(Sigma) projection_set3(Sigma,A,B,W);

%%
feasible_matrix = eye(d+k)*(sigma_squared/(1-(1-2*gamma)^2-(gamma/kappa)^2));
numIter = 10000;
stopThr = 1e-25;
lowerTriangleIndices = tril(true(d+k));
% Sigma = cov(lowerTriangleIndices);
Sigma = feasible_matrix(lowerTriangleIndices);
% Sigma = randi([-5,5],[(d+k+1)*(d+k)/2 1]);


Sigma_p = Dykstra(cProjFun,Sigma,numIter,stopThr);

%%
T_test_all = [55500:1000:60000];
ss = size(T_test_all);
num_trial = 20;
regret_all = zeros(agent_num, ss(2));

K_s = eye(k)*(-kappa)*(1e-2);
kappa_zero = 1*kappa;
gamma_zero = 1*gamma;
K_g = eye(d)*(kappa_zero);

Q_collection = rand(agent_num, d,T_test_all(ss(2)))*(C/d)*(1)+(C/d)*(0);
R_colleciton = rand(agent_num, k,T_test_all(ss(2)))*(C/k)*(1)+(C/k)*(0);

for tt=1:ss(2)
    %Online LQ Controller
    numIter = 10000;
    stopThr = 1e-25;
    
    T_test = T_test_all(tt);
    T_zero = ceil(log(T_test)*(T_test)^(2/3));
    T_one = ceil((T_test)^(1/3));
    % eta = 1/(rho^(1/4)*sqrt(T_test));
    eta = 5*1/(rho^(1/3)*(T_test)^(1/3));
    % eta = 1/(T_test)^(1/3);
    
    


    learner_cost = zeros(agent_num, num_trial,T_test);
    learner_cost_known = zeros(agent_num, num_trial,T_test);
    comparator_cost = zeros(num_trial,T_test);

    

    learner_state = mvnrnd(zeros(d,1),eye(d), agent_num)';
    learner_state_known = mvnrnd(zeros(d,1),eye(d), agent_num)';
    comparator_state = learner_state;

    learner_control = zeros(k, agent_num);
    learner_control_known = zeros(k, agent_num);
    comparator_control = zeros(k, agent_num);

    learner_state_collect = zeros(d, agent_num, T_zero);
    learner_control_collect = zeros(k, agent_num, T_zero);

    A_estimate = zeros(d, d, agent_num);
    B_estimate = zeros(d, k, agent_num);

    system_estimate = zeros(d,d+k,agent_num);
    system_estimate_pre = zeros(d, d+k, agent_num);
    system_estimate_ppre = rand(d, d+k, agent_num);
    true_system = [A B];


    Sigma = zeros((d+k+1)*(d+k)/2, agent_num);
    Sigma_known = zeros((d+k+1)*(d+k)/2, agent_num);

    diff_norm = zeros(agent_num,T_test);

    % Parameter for EXTRA
    new_P = (eye(agent_num) + P);
    P_tilde = new_P./2;

    [eigenvector, eigenvalue] = eig(P_tilde);
    lambda = sigma_squared/v_squared;
    lipschitz_contant = 2*((4*sqrt(sigma_squared)*kappa_zero/gamma_zero)^2)*(d+k*v_squared*kappa_zero^2)*log(4*T_zero)*T_zero + 2*lambda/agent_num;
%     lipschitz_contant = 4*((4*sqrt(sigma_squared)*kappa_zero/gamma_zero)^2)*(d+k*v_squared*kappa_zero^2)*log(4*T_zero)*(d+k) + 2*lambda/agent_num;
    step_size = (2*(lambda/agent_num)*min(diag(eigenvalue)))/lipschitz_contant^2;
    
    step_size = step_size*7*1e9;

    ZZ_prime_s = zeros(agent_num, d+k, d+k);
    XZ_prime_s = zeros(agent_num, d, d+k);

    mm = 1; %agent representative
    for j=1:num_trial
        disp(j)

    %     Initialize the same iterate (feasible) for all agents
        Sigma = repmat(cProjFun{1}(Sigma_p),1,agent_num);
        Sigma_known = repmat(cProjFun{1}(Sigma_p),1,agent_num);

        for i=1:T_test
            disp(i)   
            % State evolvement
            if i==1
                leanrner_state = mvnrnd(zeros(d,1),eye(d)*10, agent_num)';
                leanrner_state_known = leanrner_state;

                comparator_state = mvnrnd(zeros(d,1),eye(d)*10,1)';
            else
                noise = mvnrnd(zeros(d,1),eye(d),agent_num)';
                learner_state = A*learner_state + B*learner_control + noise;
                learner_state_known = A*learner_state_known + B*learner_control_known + noise;

                comparator_state = A*comparator_state + B*comparator_control +...
                    mvnrnd(zeros(d,1),eye(d),1)';
            end


            % Control for all agents and the comparator
            comparator_control = K_s*comparator_state;
            Sigma_buff = Sigma;
            Sigma_buff_known = Sigma_known;

            if i>T_zero && i<= T_zero + T_one
                for m=1:agent_num
                   iterate_weight_avg1 = zeros(d, d+k);
                   iterate_weight_avg2 = zeros(d, d+k);

                   for nn=1:agent_num
                      iterate_weight_avg1 = iterate_weight_avg1 + new_P(nn,m)*system_estimate_pre(:,:,nn);   
                      iterate_weight_avg2 = iterate_weight_avg2 + P_tilde(nn,m)*system_estimate_ppre(:,:,nn);
                   end    
                   gradient1 = 2*system_estimate_pre(:,:,m)*squeeze(ZZ_prime_s(m,:,:)) - 2*squeeze(XZ_prime_s(m,:,:))...
            + 2*(sigma_squared/v_squared)/agent_num*system_estimate_pre(:,:,m);

                   gradient2 = 2*system_estimate_ppre(:,:,m)*squeeze(ZZ_prime_s(m,:,:)) - 2*squeeze(XZ_prime_s(m,:,:))...
            + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,m);

                   system_estimate(:,:,m) = iterate_weight_avg1 - iterate_weight_avg2 - step_size*(gradient1 - gradient2);
                   A_estimate(:,:,m) = system_estimate(:,1:d,m);
                   B_estimate(:,:,m) = system_estimate(:,d+1:d+k,m); 
                   diff_norm(m,i) = norm(true_system - system_estimate(:,:,m), 'fro');
                end


                system_estimate_ppre = system_estimate_pre;
                system_estimate_pre = system_estimate;
            end

            % Each agent collect the data for the estimation of the system
            % matrices.
            if i<=T_zero+T_one
                learner_state_collect(:,:,i) = learner_state;
                for m=1:agent_num
                    learner_control(:,m) = mvnrnd(K_g*learner_state(:,m),2*sigma_squared*(kappa_zero^2)*eye(d),1)';


                    Q = diag(Q_collection(m, :,i));
                    R = diag(R_colleciton(m, :,i));
                    comparator_cost(j,i) = comparator_cost(j,i)+... 
                        comparator_state'*Q*comparator_state + comparator_control'*R*comparator_control;
                end
                learner_control_collect(:,:,i) = learner_control;


            else           
                for m=1:agent_num     
                    %Learner's Controller
                    y = zeros(d+k);
                    y(lowerTriangleIndices) = Sigma(:,m);
                    Sigma_matrix = y + y.' - diag(diag(y));

                    K = Sigma_matrix(d+1:d+k,1:d)/(Sigma_matrix(1:d,1:d));
                    V = Sigma_matrix(d+1:d+k,d+1:d+k) - ...
                    Sigma_matrix(d+1:d+k,1:d)*pinv(Sigma_matrix(1:d,1:d))*Sigma_matrix(d+1:d+k,1:d)';
                    %Some twisting for numerical error
                    V = (V+V')/2;
                    [U,D] = eig(V);
                    D(D<0)=0;
                    V = U*D*U';
                    V = (V+V')/2;
        %             V = eye(k)*0;
                    [eigenvector, eigenvalue] = eig(V);
                    while min(diag(eigenvalue)) < 0
                        V = V + eye(k)*(1e-15);
        %                 V = V + eye(k)*abs(min(diag(eigenvalue)));
                        [eigenvector, eigenvalue] = eig(V);
                    end
            %         fprintf('Largest singular value: %d\n', max(svds(K)));
            %         fprintf('largest eigenvalue of cov (control): %d\n',max(diag(D)));
            %         fprintf('Learner Stability:\n');
            %         disp(svds(A+B*K));
            %         disp([A B]*Sigma_matrix*[A B]'+eye(d)-Sigma_matrix(1:d,1:d))
            %         fprintf('Learner state norm: %d\n', learner_state'*learner_state);
            %         fprintf('Comparator state norm: %d\n', comparator_state'*comparator_state);



                    learner_control(:,m) = reshape(mvnrnd(K*learner_state(:,m),V),[],1);


                    Q = diag(Q_collection(m, :,i));
                    R = diag(R_colleciton(m, :,i));
                    comparator_cost(j,i) = comparator_cost(j,i)+... 
                    comparator_state'*Q*comparator_state + comparator_control'*R*comparator_control;

            %         disp([learner_cost(j,i) comparator_cost(j,i)])
            %         pause(2)

                    %Parameter update and projection
                    update = -eta*[Q zeros([d k]);zeros([k d]) R];
                    Sigma_aggregate = Sigma_buff*P(:,m);
                    Sigma(:,m) = Sigma_aggregate + update(lowerTriangleIndices);
                    cProjFun{3} = @(Sigma) projection_set3(Sigma,A_estimate(:,:,m),B_estimate(:,:,m),W);
                    Sigma(:,m) = Dykstra(cProjFun,Sigma(:,m),numIter,stopThr);
        %             Sigma(:,m) = cProjFun{1}(Sigma(:,m));

                end

            end


            cProjFun{3} = @(Sigma) projection_set3(Sigma,A,B,W);
            for m=1:agent_num     
                %Learner's Controller
                y = zeros(d+k);
                y(lowerTriangleIndices) = Sigma_known(:,m);
                Sigma_matrix = y + y.' - diag(diag(y));

                K = Sigma_matrix(d+1:d+k,1:d)/(Sigma_matrix(1:d,1:d));
                V = Sigma_matrix(d+1:d+k,d+1:d+k) - ...
                Sigma_matrix(d+1:d+k,1:d)*pinv(Sigma_matrix(1:d,1:d))*Sigma_matrix(d+1:d+k,1:d)';
                %Some twisting for numerical error
                V = (V+V')/2;
                [U,D] = eig(V);
                D(D<0)=0;
                V = U*D*U';
                V = (V+V')/2;

                [eigenvector, eigenvalue] = eig(V);
                while min(diag(eigenvalue)) < 0
                    V = V + eye(k)*(1e-15);
                    [eigenvector, eigenvalue] = eig(V);
                end



                learner_control_known(:,m) = reshape(mvnrnd(K*learner_state_known(:,m),V),[],1);


                Q = diag(Q_collection(m, :,i));
                R = diag(R_colleciton(m, :,i));


                %Parameter update and projection
                update = -eta*[Q zeros([d k]);zeros([k d]) R];
                Sigma_aggregate = Sigma_buff_known*P(:,m);
                Sigma_known(:,m) = Sigma_aggregate + update(lowerTriangleIndices);
                Sigma_known(:,m) = Dykstra(cProjFun,Sigma_known(:,m),numIter,stopThr);

            end


            if i == T_zero
                for m=1:agent_num
                   Z = [learner_state_collect(:,m,1:T_zero-1);learner_control_collect(:,m,1:T_zero-1)];
                   Z = squeeze(Z);
                   ZZ_prime_s(m,:,:) = Z*Z';
                   XZ_prime_s(m,:,:) = squeeze(learner_state_collect(:,m,2:T_zero))*Z';
                end 

                for m=1:agent_num
                    iterate_weight_avg = zeros(d, d+k);
                    for nn=1:agent_num
                       iterate_weight_avg = iterate_weight_avg + P(nn,m)*system_estimate_ppre(:,:,nn);
                    end
                    gradient = 2*system_estimate_ppre(:,:,m)*squeeze(ZZ_prime_s(m,:,:)) - 2*squeeze(XZ_prime_s(m,:,:))...
            + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,m);

                    system_estimate_pre(:,:,m) = iterate_weight_avg - step_size*gradient;
                end
            end


            for m=1:agent_num
                Q = diag(Q_collection(m, :,i));
                R = diag(R_colleciton(m, :,i));

                learner_cost(:,j,i) = learner_cost(:,j,i) + ...
                diag(learner_state'*Q*learner_state) + ...
                diag(learner_control'*R*learner_control);

                learner_cost_known(:,j,i) = learner_cost_known(:,j,i) + ...
                diag(learner_state_known'*Q*learner_state_known) + ...
                diag(learner_control_known'*R*learner_control_known);
            end    
        end

    end

    %Plot
    % cost_difference = learner_cost - comparator_cost;
    % avg_cost_difference = sum(cost_difference,1)/num_trial;
    % cost_difference_error_bar = std(cost_difference)/sqrt(num_trial);

    avg_learner_cost = sum(learner_cost,2)/num_trial;
    avg_learner_cost_known = sum(learner_cost_known,2)/num_trial;
    avg_comparator_cost = sum(comparator_cost,1)/num_trial;
    learner_cost_err_bar = std(learner_cost,0,2)/sqrt(num_trial);
    comparator_cost_err_bar = std(comparator_cost)/sqrt(num_trial);

    avg_accu_learner_cost = zeros(agent_num,T_test);
    avg_accu_learner_cost_known = zeros(agent_num,T_test);
    avg_accu_comparator_cost = zeros(1,T_test);
    % avg_accu_cost_difference = zeros(agent_num,T_test);

    avg_accu_learner_cost(:,1) = avg_learner_cost(:,1);
    avg_accu_learner_cost_known(:,1) = avg_learner_cost_known(:,1);
    avg_accu_comparator_cost(1) = avg_comparator_cost(1);
    % avg_accu_cost_difference(:,1) = avg_cost_difference(:,1);

    for i=2:T_test
        avg_accu_learner_cost(:,i) = avg_learner_cost(:,i) + avg_accu_learner_cost(:,i-1);
        avg_accu_learner_cost_known(:,i) = avg_learner_cost_known(:,i) + avg_accu_learner_cost_known(:,i-1);
        avg_accu_comparator_cost(i) = avg_comparator_cost(i) + avg_accu_comparator_cost(i-1);
    %     avg_accu_cost_difference(:,i) = avg_cost_difference(:,i) + avg_accu_cost_difference(:,i-1);
    end    

    % learner_cost = zeros(agent_num, num_trial,T_test);
    % comparator_cost = zeros(num_trial,T_test);
    accu_learner_cost = zeros(agent_num, num_trial,T_test);
    accu_comparator_cost = zeros(1,num_trial,T_test);

    accu_learner_cost(:,:,1) = learner_cost(:,:,1);
    accu_comparator_cost(:,:,1) = comparator_cost(:,1)';
    for i=2:T_test
        accu_learner_cost(:,:,i) = learner_cost(:,:,i) + accu_learner_cost(:,:,i-1);
        accu_comparator_cost(:,:,i) = comparator_cost(:,i)' + accu_comparator_cost(:,:,i-1);
    end    

    regret = (avg_accu_learner_cost-repmat(avg_accu_comparator_cost,agent_num,1));
    regret_known = (avg_accu_learner_cost_known-repmat(avg_accu_comparator_cost,agent_num,1));

    regret_all(:,tt) = regret(:,T_test);
end
%%
% regret_all = regret_all_old;
T_test_all = (500:1000:60000);
figure;
for m=1:agent_num
    plot(rregret(m,:))
%     figure;
%     plot(rregret(m,:)./(log(T_test_all).*T_test_all.^(2/3)))
%     plot(rregret(m,:)./T_test_all.^(1))
    xlabel("Time (iteration)")
    ylabel("Individual Regrets of All Agents")
%     ylabel("Individual Regrets of All Agents/(log(T)T^(^2^/^3^))")
%     ylabel("Averaged Regret over Time")
    hold on
end
hold off

%%
% regret = sum(accu_learner_cost - repmat(accu_comparator_cost, [agent_num 1 1]),2)/num_trial;
% regret_error_bar = std(accu_learner_cost - repmat(accu_comparator_cost, agent_num, 1),0,2)/sqrt(num_trial);
% 
% regret = squeeze(regret);
% regret_error_bar =  squeeze(regret_error_bar);
% indices = [1:T_test];
% regret_error_bar(:,mod(indices,1000)~=0) = 0;
% 
% T_start = 1;
% T_end = T_test;
% 
% figure;
% plot([T_start:T_end], regret(1,T_start:T_end))
% % plot([1:T_test], regret(1:T_test))
% hold on
% plot([T_start:T_end; T_start:T_end],...
%     [regret(1,T_start:T_end) - regret_error_bar(1,T_start:T_end); regret(1,T_start:T_end) + regret_error_bar(1,T_start:T_end)], '--r')
% hold off




%%

regret = (avg_accu_learner_cost-repmat(avg_accu_comparator_cost,agent_num,1));
regret_known = (avg_accu_learner_cost_known-repmat(avg_accu_comparator_cost,agent_num,1));
T_start = 1;
T_end = T_test;

figure;
for m=1:agent_num
    plot(diff_norm(m,T_zero+1:T_zero+floor(T_one)))
    hold on
end
hold off

figure;
for m=1:agent_num
    plot(regret(m,T_start:T_end))
%     xlabel("Time (iteration)")
%     ylabel("Individual Regret of Agent 1")
    hold on
end
hold off

figure;
for m=1:agent_num
    plot(regret_known(m,T_start:T_end))
%     xlabel("Time (iteration)")
%     ylabel("Individual Regret of Agent 1")
    hold on
end
hold off
%%
% T_start = T_one+T_zero;
T_start = 100;
figure;
for m=1:agent_num
    plot(regret(m,T_start:T_end)./(T_start:T_end).^(1))
%     xlabel("Time (iteration)")
%     ylabel("Individual Regret of Agent 1")
    hold on
end
hold off

figure;
for m=1:agent_num
    plot(regret_known(m,T_start:T_end)./(T_start:T_end).^(1/2))
%     xlabel("Time (iteration)")
%     ylabel("Individual Regret of Agent 1")
    hold on
end
hold off

%%

% Compute the exact minimizer
V_zero = zeros(d+k,d+k);
product_term = zeros(d,d+k);
for i=1:agent_num
    Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
    Z = squeeze(Z);
    V_zero = V_zero + (Z*Z'); 
    
    product_term = product_term + squeeze(learner_state_collect(:,i,2:T_zero))*Z';
end
V_zero  = V_zero + eye(d+k)*sigma_squared/v_squared;
system_minimizer = product_term/V_zero;

[eigenvector, eigenvalue] = eig(P_tilde);
lambda = sigma_squared/v_squared;
lipschitz_contant = 2*((4*sqrt(sigma_squared)*kappa_zero/gamma_zero)^2)*(d+k*v_squared*kappa_zero^2)*log(4*T_zero)*T_zero + 2*lambda/agent_num;
step_size = (2*lambda/agent_num*min(diag(eigenvalue)))/lipschitz_contant^2;
step_size = step_size*1e4;

% EXTRA
system_estimate = zeros(d,d+k,agent_num);
system_estimate_pre = zeros(d, d+k, agent_num);
system_estimate_ppre = rand(d, d+k, agent_num);
diff_norm = zeros(agent_num,T_test);
diff_norm2 = zeros(agent_num,T_test);

for i=1:agent_num
    iterate_weight_avg = zeros(d, d+k);
    for j=1:agent_num
       iterate_weight_avg = iterate_weight_avg + P(j,i)*system_estimate_ppre(:,:,j);
    end
    Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
    Z = squeeze(Z);
    gradient = 2*system_estimate_ppre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);
    
    
    
    system_estimate_pre(:,:,i) = iterate_weight_avg - step_size*gradient;
end

for iter=T_zero+1:T_test
   for i=1:agent_num
      iterate_weight_avg1 = zeros(d, d+k);
      iterate_weight_avg2 = zeros(d, d+k);
      
      for j=1:agent_num
         iterate_weight_avg1 = iterate_weight_avg1 + new_P(j,i)*system_estimate_pre(:,:,j);   
         iterate_weight_avg2 = iterate_weight_avg2 + P_tilde(j,i)*system_estimate_ppre(:,:,j);
      end    
      Z = [learner_state_collect(:,i,1:T_zero-1);learner_control_collect(:,i,1:T_zero-1)];
      Z = squeeze(Z);
      
      gradient1 = 2*system_estimate_pre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_pre(:,:,i);
    
      gradient2 = 2*system_estimate_ppre(:,:,i)*(Z*Z') - 2*squeeze(learner_state_collect(:,i,2:T_zero))*Z'...
        + 2*(sigma_squared/v_squared)/agent_num*system_estimate_ppre(:,:,i);
    
      system_estimate(:,:,i) = iterate_weight_avg1 - iterate_weight_avg2 - step_size*(gradient1 - gradient2);
 
   end
   
   for i=1:agent_num
       diff_norm(i,iter) =  norm(system_minimizer - system_estimate(:,:,i), 'fro');
       diff_norm2(i,iter) =  norm(true_system - system_estimate(:,:,i), 'fro');
   end
   
   system_estimate_ppre = system_estimate_pre;
   system_estimate_pre = system_estimate;
end    

%%
figure;
for m=1:agent_num
    plot(diff_norm(m,T_zero+1:T_test));
    hold on
end
hold off
%%
figure;
for m=1:agent_num
    plot(diff_norm2(m,T_zero+1:T_test));
    hold on
end
hold off