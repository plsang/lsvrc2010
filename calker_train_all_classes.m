
%% M classes, N images, R: random number
function calker_train_all_classes(M, N, R, varargin)
	
	set_env;
	
	start_class = 1;
	end_class = M;
	cv = 0;	% cross validation
	C = 1;	% C parameter for SVM
	
	for k=1:2:length(varargin),
	
		opt = lower(varargin{k});
		arg = varargin{k+1} ;
	  
		switch opt
			case 's'
				start_class = arg ;
			case 'e' ;
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
	
	ker_root_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/kernels/train';
	ker_dir = sprintf('%s/lsvrc2010_rand%dc_%di/r%d', ker_root_dir, M, N, R);
	
	fprintf('Initializing pre-computed kernels...\n');
	
	Ns = N / 2;
	train_ker = zeros(M*Ns, M*Ns);
	
	fprintf('Initializing labels...\n');
	all_labels = -ones(M, M*Ns);
	
	fprintf('Loading pre-computed kernels...\n');
	
	label_idx = 0;
	
	img_idx = [1:M*N];
	img_idx = find(mod(img_idx, N) <= Ns & mod(img_idx, N) > 0);
	
	for kk = 1:length(selected_classes),
		if mod(kk, 10) == 0, fprintf('%d ', kk); end;
		class_name = selected_classes{kk};
		
		ker_file = sprintf('%s/%s.mat', ker_dir, class_name);
		
		ker_kk = load(ker_file, 'kers');
		
		pos_idx = label_idx + [1:Ns]; 
		
		train_ker(pos_idx, :) = ker_kk.kers([1:Ns], img_idx);
		
		all_labels(kk, pos_idx) = 1;
		
		label_idx = label_idx + Ns;
	end
	
	%labels = double(labels);
	%train_ker = double(train_ker);
	
	fprintf('Start training...\n');
	for kk = start_class:end_class,
		class_name = selected_classes{kk};
	
        model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/models/%s.mat', M, N, R, class_name);
		
		if exist(model_file, 'file'),
			fprintf('Skipped training %s \n', model_file);
			continue;
		end
		
		fprintf('[%d/%d] Training class ''%s''...\n', kk - start_class + 1, end_class - end_class + 1, class_name);	
		
		labels = double(all_labels(kk,:));
		
		posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
		
		fprintf('SVM learning with predefined kernel matrix...\n');
		svm = calker_svmkernellearn(train_ker, labels,   ...
						   'type', 'C',        ...
						   ...%'C', 10,            ...
						   'verbosity', 1,     ...
						   ...%'rbf', 1,           ...
						   'crossvalidation', cv, ...
						   'C',	C, ...
						   'weights', [+1 posWeight ; -1 1]') ;
		
		svm = svmflip(svm, labels);	
		
		fprintf('\tSaving model ''%s''.\n', model_file) ;
		if ~exist(fileparts(model_file), 'file'),
			mkdir(fileparts(model_file));
		end
		save( model_file, 'svm' );	
	end
end
