function acc = ComputeAccuracy(P,y)
    %size P = KxN
    [K,N] = size(P);
    %size y = Nx1
    %To store the prediction using the P matrix.
    pred = zeros(N,1,'double');
    %Compute the result using the P matrix.
    for i = 1: N
        [argvalue, argmax] = max(P(:,i));
        pred(i,1) = argmax;
    end
    correct = 0;
    for i = 1: N
        if pred(i,1) == y(i,1)
            correct = correct + 1;
        end
    end
    acc = (correct/N)*100;
return

