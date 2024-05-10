function output = test_fun(params)
output = [];
%params.feat             = [1 1 1 1 1 2];
%params.weights          = [0.2; 0.2; 0.2; 0.2; 0.2; -1];
data                     = [-.5, 3, .2, 9, .23, 1];
%class(params)
%disp(params.weights')
%disp(size(params.weights'))
%disp(size(data * params.weights'))
output = data * params.weights';