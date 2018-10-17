function Assignment1_2
    %Now we know for sure that the gradients calculations are ok.
    %Exercise 1.2 : Training a multi-linear classifier.
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
    
    %n_batch, eta,n_epochs,lambda aprameters
    lambda = 1;
    GDparams.eta = 0.01;
    GDparams.n_batch =100;
    GDparams.n_epochs =40;
    
    %Calculate the training and validation loss. 
    Loss_validation = zeros(1,GDparams.n_epochs);
    Loss_train = zeros(1,GDparams.n_epochs);
    for j=1: GDparams.n_epochs 
        Loss_validation(j) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
        Loss_train(j) = ComputeCost(X_train, Y_train, W, b, lambda);
        [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
        W=Wstar;
        b=bstar;
    end
    %Plot the training and the validation loss.
    %Assignment description : around 2.0...
    figure()
    plot(1 : GDparams.n_epochs, Loss_train,'g')
    hold on
    plot(1 : GDparams.n_epochs, Loss_validation,'r')
    hold off
    xlabel('Epochs');
    ylabel('Loss value');
    legend('Training loss', 'Validation loss');
    
    %After training, just compute the accuracy of your learnt classifer on
    %the test data. (Assignemnt description = 36.39%)
    
    P = EvaluateClassifier(X_train,W,b);
    accuracy_train = ComputeAccuracy(P,y_train);
    disp(['Training Accuracy:' num2str(accuracy_train) '%'])
    P = EvaluateClassifier(X_test,W,b);
    accuracy_test = ComputeAccuracy(P, y_test);
    disp(['Test Accuracy:' num2str(accuracy_test) '%'])
    
    %Visualization of the weight matrix as an image. 
    K = 10;
    for i = 1 : K
        im = reshape(W(i, :), 32, 32, 3);
        s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
        s_im{i} = permute(s_im{i}, [2, 1, 3]);
    end
    figure()
    montage(s_im, 'size', [1, K])
        
    %Do the same with the following configurations:
    %lambda = 0, n_epochs = 40, n_batch = 100, eta = 0.1
    %lambda = 0, n_epochs = 40, n_batch = 100, eta =0.01
    %lambda = 0.1, n_epochs =40, n_batch = 100, eta = 0.01
    %lambda = 1, n_epochs = 40, n_batch = 100, eta = 0.01
end

