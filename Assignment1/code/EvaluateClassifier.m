function P = EvaluateClassifier(X,W,b)
    %Be sure that it is double. 
    W = double(W);
    X = double(X);
    b = double(b);
    %Softmax to generate a vector with the probability of each label.
    s = W*X + b;
    K = size(b);
    K = K(1);
    %Compute the P matrix.
    %Each column of P represents the probability of each label /image.
    P = exp(s)./((ones(1,K,'double'))*exp(s));
return

