function Assignment1_1
    %%Exercise 1.1
    %Extract data from the datafiles.
    addpath Datasets/;
    training_name = 'data_batch_1.mat';
    test_name = 'test_batch.mat';
    validation_name = 'data_batch_2.mat';
    %X,Y and y data for each file.
    [X_train,Y_train,y_train] = LoadBatch(training_name);
    [X_test,Y_test,y_test] = LoadBatch(test_name);
    [X_validation,Y_validation,y_validation] = LoadBatch(validation_name);
    
    %Initialize the parameters of the model W and b. [size(W) = 10x3072
    %size(b) = 10x1]
    %Initialize each entry to have Gaussian random values with 0 mean and
    %standard deviation 0.01.
    c = 0;
    a = 0.01;
    K = size(Y_train,1);
    d = size(X_train,1);
    b = a.*randn(K,1)+c;
    W = a.*randn(K,d)+c;
  
    %Parameters to check:
    lambda = 0;
    batch = 100;
    %Small positive number
    eps = 1e-10;
    
    %Compute the numerically gradients.
    [grad_b_n, grad_W_n] = ComputeGradsNumSlow(X_train(:, 1 : batch), Y_train(:, 1 : batch), W, b, lambda, 1e-6);
    %Generate P to be able to compute the gradients with our function
    %ComputeGradients.
    P = EvaluateClassifier(X_train(:, 1 : batch),  W, b);
    %Analytical gradients.
    [grad_W, grad_b] = ComputeGradients(X_train(:, 1 : batch), Y_train(:, 1 : batch), P,  W, lambda);
    %Compare them to be sure that the gradients are computed properly.
    %Compare them through the computation of the relative error. 
    %|ga-gn| / (max(eps,|ga|+|gn|)) where eps is a small positive number.
    %Check this is small.
    gradient_W = max(max(abs(grad_W_n - grad_W)./max(eps, abs(grad_W_n) + abs(grad_W))));
    gradient_b = max(abs(grad_b_n - grad_b)./max(eps, abs(grad_b_n) + abs(grad_b)));
    
    %Check if gradients are computed correctly.
    %Small number choosed: 1e-6 (specified in the description of the
    %assignment)
    
    %Gradient W:
    if gradient_W < 1e-6
        fprintf("Correct gradient W!");
    else
        fprintf("Incorrect gradient W!");
    end
    %Gradient b:
    if gradient_b < 1e-6
        fprintf("Correct gradient b!");
    else
        fprintf("Incorrect gradient b!");
    end    
end
