%Author: Sandra Picó Oristrell
%Data: 13 of August 2018.
%Assignment 2: 2 layer neural network.

%% Exercise 1: Read in the data and initialize the parameters of the network.
%Load the data using the function implemented in assignment 1.

[X_training, Y_training, y_training] = LoadBatch('./data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('./data_batch_2.mat');
[X_test, Y_test, y_test] = LoadBatch('./test_batch.mat');

%Apply more pre-processing to the raw input data.
%Transform it to have zero mean.
mean_X = mean(X_training, 2);
X_training = X_training - repmat(mean_X, [1, size(X_training, 2)]);
X_validation   = X_validation   - repmat(mean_X, [1, size(X_validation,   2)]);
X_test  = X_test  - repmat(mean_X, [1, size(X_test,  2)]);

%Variables
d = size(X_training,1);
N = size(X_training,2);
K = size(Y_training,1);

%Data structure for the parameters of the network and initialize the
%values.

%50 nodes in the hidden layer
m = 50;
lambda = 0;
%Initialize the network:
[W,b] = Initialize_Network(m,d,K);

%% Exercise 2: Compute the gradients for the network parameters.

%Re-write or update gradients functions from assignment 1.
%% 2.1 Check that the gradients are implemented properly.

batch_size = 100;

[P,h] = EvaluateClassifier(X_training(:,1:batch_size),W,b);
[grad_W, grad_b] = ComputeGradients(X_training(:,1:batch_size),Y_training(:,1:batch_size),P,h,W,b,lambda);
[ngrad_W, ngrad_b] = ComputeGradsNumSlow(X_training(:,1:batch_size),Y_training(:,1:batch_size),W,b,lambda,1e-5);

%Check the comparison between the numerical and the computed gradients:
eps = 1e-10;
%Layer 1: 
gradient_b1_comparison = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
gradient_W1_comparison = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
%Layer 2:
gradient_b2_comparison = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
gradient_W2_comparison = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));

%Check that the gradients have small number.
fprintf("Results for the calculated gradients:");
fprintf("Layer 1:");
fprintf("W1: %f",gradient_W1_comparison);
fprintf("b1: %f",gradient_b1_comparison);
fprintf("Layer 2:");
fprintf("W2: %f",gradient_W2_comparison);
fprintf("b2: %f",gradient_b2_comparison);

%% 2.2 Try if you can overfit in training data with 200 epochs and reasonable learning rate.

GDparams.eta = 0.01;
GDparams.n_batch = 1;
GDparams.n_epochs = 300;
lambda = 0;

%Save all the information per each epoch.
cost_training_list = zeros(1, GDparams.n_epochs);
cost_validation_list = zeros(1, GDparams.n_epochs);
accuracy_training_list = zeros(1, GDparams.n_epochs);
accuracy_validation_list = zeros(1, GDparams.n_epochs);
epochs_list = zeros(1, GDparams.n_epochs);

%Try with only 100 examples.
examples = 100;
X_train = X_training(:,1:examples);
Y_train = Y_training(:,1:examples);
y_train = y_training(:,1:examples);

X_val = X_validation(:,1:examples);
Y_val = Y_validation(:,1:examples);
y_val = y_validation(:,1:examples);

for i = 1:GDparams.n_epochs
    [W,b] = MiniBatchGD(X_train, Y_train,y_train,GDparams,W,b,lambda);
    fprintf("Epoch: %d\n",i);
    epochs_list(i) = i;
    cost_training_list(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    accuracy_training_list(i) = ComputeAccuracy(X_train,y_train,W,b);
    accuracy_validation_list(i) = ComputeAccuracy(X_val,y_val,W,b);
    cost_validation_list(i) = ComputeCost(X_val, Y_val, W, b,lambda);
end

%Plot cost in training and validation per epoch.
title_text = "Cost in validation and training dataset";
PlotCost(epochs_list, cost_validation_list, cost_training_list,title_text);

%Plot accuracy in training and validation per epoch.
title_text = "Accuracy in validation and training dataset";
PlotAccuracy(epochs_list,accuracy_validation_list, accuracy_training_list,title_text);

%Final accuracy in validation dataset and training dataset:
accuracy_final_training = ComputeAccuracy(X_train,y_train,W,b);
fprintf("Final training accuracy: %f %", accuracy_final_training);

accuracy_final_validation = ComputeAccuracy(X_val,y_val,W,b);
fprintf("Final validation accuracy: %f %",accuracy_final_validation);

%%  Exercise 3: Add momentum to your update step.
%To help speed up training times you should add momentum terms into your mini-batch update steps

GDparams.eta = 0.001;
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
lambda = 0;

eta_decay_rate = 0.95;

%Compare rho values = [0.5,0.9,0.99];
rho_values = [0.50,0.9,0.99];

for rho = rho_values
    [W,b] = Initialize_Network(m,d,K);
    GDparams.eta = 0.01;
    cost_training_list = zeros(1, GDparams.n_epochs);
    cost_validation_list = zeros(1, GDparams.n_epochs);
    accuracy_training_list = zeros(1, GDparams.n_epochs);
    accuracy_validation_list = zeros(1, GDparams.n_epochs);
    epochs_list = zeros(1, GDparams.n_epochs);
    for i = 1:GDparams.n_epochs
        [W,b] = MiniBatchGD_withMomentum(X_training, Y_training,y_training,GDparams,W,b,lambda,rho);
        fprintf("Epoch: %d\n",i);
        epochs_list(i) = i;
        cost_training_list(i) = ComputeCost(X_training, Y_training, W, b, lambda);
        accuracy_training_list(i) = ComputeAccuracy(X_training,y_training,W,b);
        accuracy_validation_list(i) = ComputeAccuracy(X_validation,y_validation,W,b);
        cost_validation_list(i) = ComputeCost(X_validation, Y_validation, W, b,lambda);
        GDparams.eta  = GDparams.eta * eta_decay_rate;
    end
    %Plot cost in training and validation per epoch.
    title_text = "Cost validation/training with rho = " + num2str(rho);
    PlotCost(epochs_list, cost_validation_list, cost_training_list,title_text);
    %Plot accuracy in training and validation per epoch.
    title_text = "Accuracy validation/training with rho = " + num2str(rho);
    PlotAccuracy(epochs_list,accuracy_validation_list, accuracy_training_list,title_text);
    %Final accuracy in validation dataset and training dataset:
    accuracy_final_training = ComputeAccuracy(X_training,y_training,W,b);
    fprintf("Final training accuracy: %f %", accuracy_final_training);
    accuracy_final_validation = ComputeAccuracy(X_validation,y_validation,W,b);
    fprintf("Final validation accuracy: %f %",accuracy_final_validation);
end

%% Exercise 4: Training your network
% All the experiments will be using all the examples.

%%  4.1 Find reasonable values for the learning rate:


%Regularization term to small value: 
lambda = 0.000001;
rho = 0.9;
eta_values = [0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001];
GDparams.n_batch = 100;
GDparams.n_epochs = 10;

final_cost_training_list = {};
for eta = eta_values
    GDparams.eta = eta;
    cost_training_list = zeros(1, GDparams.n_epochs);
    [W,b] = Initialize_Network(m,d,K);
    fprintf("Eta value: %f\n",eta);
    for i = 1:GDparams.n_epochs
        fprintf("Epoch: %d\n",i);
        [W,b] = MiniBatchGD_withMomentum(X_training, Y_training,y_training,GDparams,W,b,lambda,rho);    
        cost_training_list(i) = ComputeCost(X_training, Y_training,W,b,lambda);
    end
    final_cost_training_list{end+1} = cost_training_list;
end
epochs_list = zeros(1,GDparams.n_epochs);
for i = 1: GDparams.n_epochs
    epochs_list(i) = i;
end
%Plot the cost per each case of eta.
PlotCost_Eta_values(epochs_list, eta_values, final_cost_training_list);

%% 4.2 Coarse-search random.

GDparams.n_batch = 100;
GDparams.n_epochs = 10;
decay_rate= 0.95;
rho=0.9;
n_pairs= 70;
%Learning rate range:
eta_max = 0.1;
eta_min = 0.001;
%Lambda range:
lambda_max = 0.1;
lambda_min = 1e-7;

validation_accuracy_list = zeros(1,n_pairs);
eta_values_list = zeros(1,n_pairs);
lambda_values_list = zeros(1,n_pairs);

for j = 1:n_pairs
    fprintf("\n Pair number: %d \n ", j);
    eta_exp = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
    eta = 10^eta_exp;
    lambda_exp = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
    lambda = 10^lambda_exp;
    
    GDparams.eta = eta;    
    [W,b] = Initialize_Network(m,d,K);
    
    for i=1: GDparams.n_epochs
        fprintf("Epoch: %d\n", i);
        [W, b] = MiniBatchGD_withMomentum(X_training, Y_training,y_training, GDparams, W, b, lambda, rho);
        GDparams.eta = GDparams.eta*decay_rate;
    end
    validation_accuracy_list(j) = ComputeAccuracy(X_validation,y_validation,W,b);
    eta_values_list(j) = GDparams.eta;
    lambda_values_list(j) = lambda; 
end

%Sort the array:
[validation_accuracy_list,validation_indexs] = sort(validation_accuracy_list,'descend');
eta_values_list = eta_values_list(validation_indexs);
lambda_values_list = lambda_values_list(validation_indexs);

%Write file: 
filename = 'exercise_4.2_coarse_search.txt';
fid = fopen(filename,'wt');
fprintf(fid,"\nAccuracy values:\n");
fprintf(fid,'%f\t',validation_accuracy_list);  
fprintf(fid,"\nEta values:\n");
fprintf(fid,'%f\t',eta_values_list);
fprintf(fid,"\nLambda values:\n");
fprintf(fid,'%f\t',lambda_values_list); 
fprintf(fid,"\n");
fclose(fid);

%% 4.2 Coarse-fine search random.
%Better values for lambda search.

GDparams.n_batch = 100;
GDparams.n_epochs = 10;
decay_rate= 0.95;
rho=0.9;
n_pairs= 70;
%Learning rate range:
eta_max = 0.07;
eta_min = 0.01;
%Lambda range:
lambda_max = 0.005;
lambda_min = 1e-6;

validation_accuracy_list = zeros(1,n_pairs);
eta_values_list = zeros(1,n_pairs);
lambda_values_list = zeros(1,n_pairs);

for j = 1:n_pairs
    fprintf("\n Pair number: %d \n ", j);
    eta_exp = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
    eta = 10^eta_exp;
    lambda_exp = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
    lambda = 10^lambda_exp;
    
    GDparams.eta = eta;    
    [W,b] = Initialize_Network(m,d,K);
    
    for i=1: GDparams.n_epochs
        fprintf("Epoch: %d\n", i);
        [W, b] = MiniBatchGD_withMomentum(X_training, Y_training,y_training, GDparams, W, b, lambda, rho);
        GDparams.eta = GDparams.eta*decay_rate;
    end
    validation_accuracy_list(j) = ComputeAccuracy(X_validation,y_validation,W,b);
    eta_values_list(j) = GDparams.eta;
    lambda_values_list(j) = lambda; 
end

%Sort the array:
[validation_accuracy_list,validation_indexs] = sort(validation_accuracy_list,'descend');
eta_values_list = eta_values_list(validation_indexs);
lambda_values_list = lambda_values_list(validation_indexs);

%Write file: 
filename = 'exercise_4.2_coarse_fine_search.txt';
fid = fopen(filename,'wt');
fprintf(fid,"\nAccuracy values:\n");
fprintf(fid,'%f\t',validation_accuracy_list);  
fprintf(fid,"\nEta values:\n");
fprintf(fid,'%f\t',eta_values_list);
fprintf(fid,"\nLambda values:\n");
fprintf(fid,'%f\t',lambda_values_list); 
fprintf(fid,"\n");
fclose(fid);

%% 4.3 Best hyper-parameters for training the network.
%Best combination

m= 50;

%Train with all the training data:
[X_training1, Y_training1, y_training1] = LoadBatch('./data_batch_1.mat');
[X_training2, Y_training2, y_training2] = LoadBatch('./data_batch_2.mat');
[X_training3, Y_training3, y_training3] = LoadBatch('./data_batch_3.mat');
[X_training4, Y_training4, y_training4] = LoadBatch('./data_batch_4.mat');
[X_training5, Y_training5, y_training5] = LoadBatch('./data_batch_5.mat');

%Test dataset
[X_test, Y_test, y_test] = LoadBatch('./test_batch.mat');

%except 1000 samples for validation: using the validation dataset
%(data_batch_2.mat)
X_validation = X_training2(:, 1:1000);
Y_validation = Y_training2(:, 1:1000);
y_validation = y_training2(:, 1:1000);
X_training2 = X_training2(:, 1001:10000);
Y_training2 = Y_training2(:, 1001:10000);
y_training2 = y_training2(:, 1001:10000);

%All training data together:
X_training = [X_training1, X_training2, X_training3];
Y_training = [Y_training1, Y_training2, Y_training3];
y_training = [y_training1, y_training2, y_training3];

%Pre-processing:
mean_X = mean(X_training, 2);
X_training = X_training - repmat(mean_X, [1, size(X_training, 2)]);
X_validation   = X_validation   - repmat(mean_X, [1, size(X_validation,   2)]);
X_test  = X_test  - repmat(mean_X, [1, size(X_test,  2)]);

%Variables
d = size(X_training,1);
N = size(X_training,2);
K = size(Y_training,1);

GDparams.n_batch = 100;
GDparams.n_epochs = 30;
decay_rate= 0.95;
rho= 0.9;
%Best combination:
GDparams.eta = 0.014001;
lambda = 0.003347;

%Plot the training and validation cost after each epoch of training 
cost_training_list = zeros(1, GDparams.n_epochs);
cost_validation_list = zeros(1, GDparams.n_epochs);
[W,b] = Initialize_Network(m,d,K);
%Original training cost
original_training_cost = ComputeCost(X_training, Y_training,W,b,lambda);
for i=1:GDparams.n_epochs
    [W, b] = MiniBatchGD_withMomentum(X_training, Y_training,y_training, GDparams, W, b, lambda, rho);
    cost_training_list(i) = ComputeCost(X_training, Y_training,W,b,lambda);
    cost_validation_list(i) = ComputeCost(X_validation, Y_validation,W,b,lambda);
    GDparams.eta = GDparams.eta*decay_rate;
    if cost_training_list(i) > 3*original_training_cost
        fprintf("Cost_training(i) > 3*original_training_cost");
        i = GDparams.n_epochs;
    end
end

epochs_list = zeros(1, GDparams.n_epochs);
for i=1:GDparams.n_epochs
   epochs_list(i) = i; 
end

%Plot the loss per epochs in validation and in training:
title_text = "Validation/Training cost per epoch";
PlotCost(epochs_list, cost_validation_list, cost_training_list,title_text)

%Final accuracy in testing and training:
final_accuracy_test = ComputeAccuracy(X_test,y_test,W,b);
fprintf("Final accuracy in test: %f % \n", final_accuracy_test);
final_accuracy_training = ComputeAccuracy(X_training,y_training,W,b);
fprintf("Final accuracy in training: %f % \n", final_accuracy_training);

%% Functions implementation


function PlotCost_Eta_values(epochs_list, eta_values, final_cost_training_list)
    figure;
    for i = 1:length(eta_values)
        text_label = "eta = " + num2str(eta_values(i));
        plot(epochs_list,final_cost_training_list{i},'DisplayName',text_label);
        hold on;
    end
    hold off;
    xlabel('Epochs');
    ylabel('Cost');
    legend;
end 

%Plot accuracy for validation/test - training dataset per epoch.
function PlotAccuracy(epochs, accuracy_validation, accuracy_training,title_text)
    figure;
    plot(epochs,accuracy_validation,epochs,accuracy_training);
    title(title_text);
    xlabel('Epochs');
    ylabel('Accuracy %');
    legend('Accuracy in validation', 'Accuracy in training','Location','southwest');
end

%Plot the loss/cost for validation/test -training dataset per epoch.
function PlotCost(epochs, cost_validation, cost_training,title_text)
    figure;
    plot(epochs,cost_validation,epochs,cost_training);
    title(title_text);
    xlabel('Epochs');
    ylabel('Cost');
    legend('Cost in validation', 'Cost in training','Location','southwest');  
end

%Function to initialize the values of the Weight matrices.
function [W,b] = Initialize_Network(m,d,K)
    W1 = randn(m,d)*0.001;
    W2 = randn(K,m)*0.001;
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    W = {W1,W2};
    b = {b1, b2};
end

function [X,Y,y] = LoadBatch(filename)
    A = load(filename);
    X = double(A.data');
    X= X/255; 
    y = double(A.labels') + 1;
    vec = ind2vec(y);
    Y = full(vec);      %One-hot representation
end

%Function to compute the cost of the 2-layer network.
function J = ComputeCost(X,Y,W,b,lambda)
    [P,h] = EvaluateClassifier(X,W,b);
    crossentropy_term = sum(diag(-log(double(Y)'*P)));
    reg_term = sum(W{1}(:).^2)+sum(W{2}(:).^2);
    J = (1/(size(X,2))*crossentropy_term)+(lambda*reg_term);     
end

function [P,h] = EvaluateClassifier(X,W,b)
    s1 = W{1}*X + b{1};
    h = max(0, s1);
    s = W{2} * h + b{2};
    P = exp(s)./sum(exp(s));
end

%Function to compute the accuracy in a 2-layer neural network:
function acc = ComputeAccuracy(X,y,W,b)
    [P,h] = EvaluateClassifier(X, W, b);
    [~,pred_idx] = max(P);
    acc=((sum(pred_idx==y)/size(X,2)*100));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W,b,lambda)
    k_x = size(X,2);
    W1 = cell2mat(W(1));
    b1 = cell2mat(b(1));
    W2 = cell2mat(W(2));
    b2 = cell2mat(b(2));   
    grad_W1 = zeros(size(W1));
    grad_W2 = zeros(size(W2));
    grad_b1 = zeros(size(b1));
    grad_b2 = zeros(size(b2));
    
    for i= 1:k_x
        P_i = P(:, i);
        h_i = h(:, i);
        Y_i = Y(:, i);
        X_i = X(:, i);        
        g = -(Y_i-P_i)';
        grad_b2 = grad_b2 + g';
        grad_W2 = grad_W2 + g'*h_i';
        
        h_i(find(h_i > 0)) = 1;
        g = g*W2*diag(h_i);
        
        grad_b1 = grad_b1 + g';
        grad_W1 = grad_W1 + g'*X_i';   
    end
    grad_b1 = grad_b1/k_x;
    grad_W1 = grad_W1/k_x + 2*lambda*W1;
    grad_b2 = grad_b2/k_x;
    grad_W2 = grad_W2/k_x + 2*lambda*W2;
    
    grad_b = {grad_b1, grad_b2};
    grad_W = {grad_W1, grad_W2}; 
end

function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)
    grad_W = cell(numel(W), 1);
    grad_b = cell(numel(b), 1);
    for j=1:length(b)
        grad_b{j} = zeros(size(b{j}));
        for i=1:length(b{j})
            b_try = b;
            b_try{j}(i) = b_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W, b_try, lambda);
            b_try = b;
            b_try{j}(i) = b_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W, b_try, lambda);
            grad_b{j}(i) = (c2-c1) / (2*h);
        end
    end
    for j=1:length(W)
        grad_W{j} = zeros(size(W{j}));
        for i=1:numel(W{j})
            W_try = W;
            W_try{j}(i) = W_try{j}(i) - h;
            c1 = ComputeCost(X, Y, W_try, b, lambda);
            W_try = W;
            W_try{j}(i) = W_try{j}(i) + h;
            c2 = ComputeCost(X, Y, W_try, b, lambda);
            grad_W{j}(i) = (c2-c1) / (2*h);
        end
    end
end


%Mini batch function implementation without momentum and learning rate
%decay.
function [Wstar, bstar] = MiniBatchGD(X,Y,y,GDparams,W,b,lambda)   
    N = size(X,2);
    eta = GDparams.eta;
    n_batch = GDparams.n_batch;

    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        indx = j_start:j_end;
        Xbatch = X(:,indx); 
        Ybatch = Y(:,indx);
        [P,h] = EvaluateClassifier(Xbatch,W,b);
        [grad_W, grad_b] = ComputeGradients(Xbatch,Ybatch,P,h, W,b,lambda);
        W{1} = W{1} - eta*grad_W{1};
        W{2} = W{2} - eta*grad_W{2};
        b{1} = b{1} - eta*grad_b{1};
        b{2} = b{2} - eta*grad_b{2};
    end       
    bstar = b;
    Wstar = W;
end

function [v_W, v_b] = InitializeMomentum(W,b)
    v_b = {zeros(size(b{1})), zeros(size(b{2}))};
    v_W = {zeros(size(W{1})), zeros(size(W{2}))};
end

function [Wstar, bstar] = MiniBatchGD_withMomentum(X, Y,y,GDparams,W,b,lambda,rho)
    N = size(X,2);
    eta = GDparams.eta;
    n_batch = GDparams.n_batch;
    [v_W,v_b] = InitializeMomentum(W,b);
    
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        indx = j_start:j_end;
        Xbatch = X(:,indx); 
        Ybatch = Y(:,indx);
        [P,h] = EvaluateClassifier(Xbatch,W,b);
        [grad_W, grad_b] = ComputeGradients(Xbatch,Ybatch,P,h, W,b,lambda);
        %Update with momentum:
        v_W{1} = rho*v_W{1} + eta*grad_W{1};
        v_b{1} = rho*v_b{1} + eta*grad_b{1};
        v_W{2} = rho*v_W{2} + eta*grad_W{2};
        v_b{2} = rho*v_b{2} + eta*grad_b{2};
        W{1} = W{1} - v_W{1};
        b{1}  = b{1} - v_b{1};
        W{2} = W{2} - v_W{2};
        b{2}  = b{2} -  v_b{2};
    end       
    bstar = b;
    Wstar = W;
end





