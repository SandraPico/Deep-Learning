function [grad_W,grad_b] = ComputeGradients(X,Y,P,W,lambda)  
    % grad_W: gradient matrix of the cost J respect to W.
    % grad_b: gradient vector of the cost J respect to b. 

    %Initialize the gradients with the respective sizes.
    grad_W = zeros(size(W));
    grad_b = zeros(size(W, 1), 1);

    %Compute gradients from the slides (lectures). 
    for i = 1 : size(X, 2)
        Xi = X(:, i);   
        Pi = P(:, i);
        Yi = Y(:, i);     
        g = -Yi'/(Yi'*Pi)*(diag(Pi) - Pi*Pi'); 
        grad_b = grad_b + g';
        grad_W = grad_W + g'*Xi';
    end
    
    %Normalize the gradients.
    grad_b = grad_b/size(X, 2);
    grad_W = grad_W/size(X, 2) + (2*lambda)*W;    
end

