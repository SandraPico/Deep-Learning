function [Wstar,bstar] = MiniBatchGD(X,Y,GDparams,W,b,lambda)
    %Size of X is : dxN
    N = size(X, 2);
    
    %GDparams is an object containing the parameter values n_batch, eta and
    %n_epochs.
    eta = GDparams.eta;
    n_batch = GDparams.n_batch;
    GDparams.n_epochs;
    
    %Structure of the assignment description.
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);  
        %EvaluateClassifier to be able to compute the gradients at each
        %mini-batch computation.
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        %Update the values based with the learning-rate.
        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
    %Parameters that we would like to return
    Wstar = W;
    bstar = b;
end

