function J = ComputeCost(X,Y,W,b,lambda)
    %Size values
    K = 10;
    %In this case, N will represent the D subset.
    [d,N] = size(X);
       
    %Compute P
    W = double(W);
    X = double(X);
    b = double(b);
    s = W*X + b;
    K = size(b);
    K = K(1);
    P = exp(s)./((ones(1,K,'double'))*exp(s));
    
    %Cross-Entropy loss
    crossEntropy_term = 0;
    for i = 1:N
        y = Y(:,i);
        p = P(:,i);
        py = y.'*p;
        crossEntropy_term = crossEntropy_term -log(py);
    end
    
    %Regularization term.
    reg_term = 0;
    for row = 1:K
        for column = 1:d
            reg_term = reg_term + W(row,column)^2;
        end
    end
    reg_term = lambda*reg_term;
    
    %Calculate J. 
    J = 1./abs(N)*crossEntropy_term + reg_term;
    
return

