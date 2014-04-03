
%% M classes, N images, R: random number
function calker_train_all_classes(M, N, R, varargin)
	
	set_env;
	
	start_class = 1;
	end_class = M;
	cv = 0;	% cross validation
	C = 1;	% C parameter for SVM
	MaxNeg = 10000; % max negative
	
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
			case 'maxneg'
				MaxNeg = arg ;
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
	

	train_ker = zeros(M*N, M*N);
	
	fprintf('Initializing labels...\n');
	all_labels = -ones(M, M*N);
	
	fprintf('Assembling pre-computed kernels...\n');
	
	label_idx = 0;
	
	%img_idx = [1:M*N];
	%img_idx = find(mod(img_idx, N) <= Ns & mod(img_idx, N) > 0);
	
	for kk = 1:length(selected_classes),
		if mod(kk, 10) == 0, fprintf('%d ', kk); end;
		class_name = selected_classes{kk};
		
		ker_file = sprintf('%s/%s.mat', ker_dir, class_name);
		
		ker_kk = load(ker_file, 'kers');
		
		pos_idx = label_idx + [1:N]; 
		
		train_ker(pos_idx, :) = ker_kk.kers;
		
		all_labels(kk, pos_idx) = 1;
		
		label_idx = label_idx + N;
	end

	
	fprintf('Removing all-zero indexes...\n');
	nonzero_idx = any(train_ker, 1);
	train_ker = train_ker(nonzero_idx, nonzero_idx); 

	
	%labels = double(labels);
	%train_ker = double(train_ker);
	
	fprintf('Start training...\n');
	for kk = start_class:end_class,
		class_name = selected_classes{kk};
	
        model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/models-%d/%s.mat', M, N, R, MaxNeg, class_name);
		
		if exist(model_file, 'file'),
			fprintf('Skipped training %s \n', model_file);
			continue;
		end
		
		fprintf('[%d/%d] Training class ''%s''...\n', kk - start_class + 1, end_class - start_class + 1, class_name);	
		
		labels = double(all_labels(kk,:));
		labels = labels(nonzero_idx);	% Removing all-zero indexes
		
		fprintf('SVM learning with predefined kernel matrix...\n');
		
		if MaxNeg ~= 0, %% subsample negative
			neg_idx = find(labels == -1);
			pos_idx = find(labels == 1);
			
			ridx = randperm(length(neg_idx));
			r_neg_idx = neg_idx(ridx(1:MaxNeg));
			r_train_idx = [pos_idx, r_neg_idx];
			
			%posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
			posWeight = ceil(length(r_neg_idx)/length(pos_idx));
						
			svm = calker_svmkernellearn(train_ker(r_train_idx, r_train_idx), labels(r_train_idx),   ...
							   'type', 'C',        ...
							   ...%'C', 10,            ...
							   'verbosity', 1,     ...
							   ...%'rbf', 1,           ...
							   'crossvalidation', cv, ...
							   'C',	C, ...
							   'weights', [+1 posWeight ; -1 1]') ;
		else	%% all negatives
			posWeight = ceil(length(find(labels == -1))/length(find(labels == 1)));
			
			svm = calker_svmkernellearn(train_ker, labels,   ...
							   'type', 'C',        ...
							   ...%'C', 10,            ...
							   'verbosity', 1,     ...
							   ...%'rbf', 1,           ...
							   'crossvalidation', cv, ...
							   'C',	C, ...
							   'weights', [+1 posWeight ; -1 1]') ;
		end
		
		svm = svmflip(svm, labels(r_train_idx));	
		
		fprintf('\tSaving model ''%s''.\n', model_file) ;
		if ~exist(fileparts(model_file), 'file'),
			mkdir(fileparts(model_file));
		end
		save( model_file, 'svm' );	
	end
end
