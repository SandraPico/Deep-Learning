function[X,Y,y] = LoadBatch(filename)
    %Open the dataset 
    addpath Datasets/;
    dataset = load(filename);
    %X contains the image pixel data. Size 3072x10000. Double type between
    %0 and 1.
    %Reshape the pixel data into a 3072x10000 array
    I = reshape(dataset.data.',32,32,3,10000);
    I = permute(I,[2,1,3,4]);
    X = reshape(I,[3072,10000]);
    %Normalize the vector
    X = double(X);
    X = X./255.0;
    
    %y is a vector of 10000 containing the label for each image.
    %better to encode it between 1-10 instead of 0-9.
    y = dataset.labels;
    for i = 1:10000
        y(i,1) = y(i,1) + 1;
    end

    %Y is a 10x10000 array with the one_hot representation.
    %Extract the labels dataset
    Y = dataset.labels;
    %One-hot representation
    num_labels = 10;     %Number of labels
    Y_one_hot = zeros(size(Y,1),num_labels);
    Y_one_hot = reshape(Y_one_hot,[10,10000]); %Reshape the one_hot representation.
    for i = 1:10000
        row = Y_one_hot(:,i);
        label = Y(i,:);
        row(label+1) = 1;
        Y_one_hot(:,i) = row;
    end
    Y = Y_one_hot;
    
return

