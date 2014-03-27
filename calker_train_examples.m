
%% M classes, N images, R: random number
function calker_train_examples(M, N, R, varargin)
	
	test_on_train = 0;
	start_class = 1;
	end_class = M;
	cv = 0;	% cross validation
	C = 1;	% C parameter for SVM
	
	for k=1:2:length(varargin),
	
		opt = lower(varargin{k});
		arg = varargin{k+1} ;
	  
		switch opt
			case 'test_on_train' 
				test_on_train = arg ;
			case 'start_class'
				start_class = arg ;
			case 'end_class' ;
				end_class = arg ;
			case 'cv' ;
				cv = arg ;
			case 'C' ;
				C = arg ;
			otherwise
				error(sprintf('Option ''%s'' unknown.', opt)) ;
		end  
	end

	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	if ~exist(imdb_file, 'file'),
		error();
	end
	
	fprintf('Loading selected image db...\n');
	imdb = load(imdb_file, 'imdb');
	imdb = imdb.imdb;
	
	selected_classes = fieldnames(imdb);
	
	hists_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/hists.mat', M, N, R);
	labels_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/labels.mat', M, N, R);
	train_ker_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/train_ker.mat', M, N, R);
	
	if ~exist(hists_file, 'file') && exist(labels_file, 'file'),
		error();
	end
	
	fprintf('Loading hists...\n');
	hists = load(hists_file, 'hists');
	labels = load(labels_file, 'labels');
	
	hists = hists.hists;
	labels = labels.labels;
		
	if ~exist(train_ker_file, 'file'),	
		error();
	end

	fprintf('Loading pre-computed kernels...\n');
	train_ker = load(train_ker_file, 'train_ker');
	train_ker = train_ker.train_ker;
	
	labels = double(labels);
	train_ker = double(train_ker);
	
	fprintf('Preparing labels...\n');
	all_labels = zeros(M, length(labels));

    for ii = 1:length(labels),
        for jj = 1:M,
            if labels(ii) == jj,
                all_labels(jj, ii) = 1;
            else
                all_labels(jj, ii) = -1;
            end
        end
    end
	
	fprintf('start training...\n');
	for kk = start_class:end_class,
		class_name = selected_classes{kk};
	
        model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/models/%s.mat', M, N, R, class_name);
		
		if exist(model_file, 'file'),
			fprintf('Skipped training %s \n', model_file);
			continue;
		end
		
		fprintf('Training class ''%s''...\n', class_name);	
		
		labels = double(all_labels(kk,:));
		posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
		
		fprintf('SVM learning with predefined kernel matrix...\n');
		svm = calker_svmkernellearn(train_ker, labels,   ...
						   'type', 'C',        ...
						   ...%'C', 10,            ...
						   'verbosity', 0,     ...
						   ...%'rbf', 1,           ...
						   'crossvalidation', cv, ...
						   'C',	C, ...
						   'weights', [+1 posWeight ; -1 1]') ;
		
		svm = svmflip(svm, labels);	
		
		if test_on_train, % test_on_train
			
			if ~exist(hists_file, 'file') && exist(labels_file, 'file'),
				error();
			end
	
			fprintf('Loading hists...\n');
			hists = load(hists_file, 'hists');
			hists = hists.hists;
	
			scores = svm.alphay' * hists(svm.svind, :) + svm.b ;
			errs = scores .* labels < 0 ;
			err  = mean(errs) ;
			selPos = find(labels > 0) ;
			selNeg = find(labels < 0) ;
			werr = sum(errs(selPos)) * posWeight + sum(errs(selNeg)) ;
			werr = werr / (length(selPos) * posWeight + length(selNeg)) ;
			fprintf('\tSVM training error: %.2f%% (weighed: %.2f%%).\n', ...
			  err*100, werr*100) ;
			fprintf('\tNumber of support vectors: %d\n', length(svm.svind)) ;
			
		end
		
		fprintf('\tSaving model ''%s''.\n', model_file) ;
		if ~exist(fileparts(model_file), 'file'),
			mkdir(fileparts(model_file));
		end
		save( model_file, 'svm' );	
	end
end