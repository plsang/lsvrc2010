
%% M classes, N images, R: random number
function calker_train_all_classes(M, N, R, varargin)
	
	set_env;
	
	start_class = 1;
	end_class = M;
	cv = 0;	% cross validation
	C = 1;	% C parameter for SVM
	MaxNeg = 10000; % max negative
	fea_pat = 'covdet.hessian.sift.cb256.pca80.fisher';
	
	rand_ann = 5;
	
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
			case 'randann'
				rand_ann = arg;
			case 'fea'
				fea_pat = arg;	
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
	ker_dir = sprintf('%s/lsvrc2010_rand%dc_%di/%s/r%d', ker_root_dir, M, N, fea_pat, R);
	
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
	
	fprintf('Start training...\n');
	for kk = start_class:end_class,
		class_name = selected_classes{kk};
		
		fprintf('[%d/%d] Training class ''%s''...\n', kk - start_class + 1, end_class - start_class + 1, class_name);	
		
		labels = double(all_labels(kk,:));
		labels = labels(nonzero_idx);	% Removing all-zero indexes
		
		labeled_model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_M%d_N%d_R%d/%s/models/%s.mat', M, N, R, fea_pat, class_name);
		fprintf('Learning labelled models...\n');
		
		% assume MaxNeg ~= 0
		if ~exist(labeled_model_file, 'file'),
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
			
			svm = svmflip(svm, labels(r_train_idx));	
		
			fprintf('\tSaving labelled model ''%s''.\n', labeled_model_file) ;
			if ~exist(fileparts(labeled_model_file), 'file'),
				mkdir(fileparts(labeled_model_file));
			end
			save( labeled_model_file, 'svm' );	
		
		else
			fprintf('Skipped training %s \n', labeled_model_file);
		end
		
		fprintf('Learning random unlabelled models...\n');
		
		for rr=1:rand_ann,
			rand_model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_M%d_N%d_R%d/%s/models-r%d/%s.mat', M, N, R, fea_pat, rr, class_name);
			
			fprintf(' --- [%d/%d] Learning random unlabelled models...\n', rr, rand_ann);
			
			if exist(rand_model_file, 'file'),
				fprintf('Skipped training %s \n', rand_model_file);
				continue;
			end
			
			pos_idx = find(labels == 1);
			
			r_pos_idx = randperm(length(labels));
			r_pos_idx = r_pos_idx(1:length(pos_idx));
			
			r_neg_idx = setdiff(1:length(labels), r_pos_idx);
			ridx = randperm(length(r_neg_idx));
			r_neg_idx = r_neg_idx(ridx(1:MaxNeg));
			
			r_train_idx = [r_pos_idx, r_neg_idx];
			
			labels(r_pos_idx) = 1;
			labels(r_neg_idx) = -1;
			
			posWeight = ceil(length(r_neg_idx)/length(pos_idx));
						
			svm = calker_svmkernellearn(train_ker(r_train_idx, r_train_idx), labels(r_train_idx),   ...
							   'type', 'C',        ...
							   ...%'C', 10,            ...
							   'verbosity', 1,     ...
							   ...%'rbf', 1,           ...
							   'crossvalidation', cv, ...
							   'C',	C, ...
							   'weights', [+1 posWeight ; -1 1]') ;
		
			svm = svmflip(svm, labels(r_train_idx));	
			
			fprintf('\tSaving model ''%s''.\n', rand_model_file) ;
			if ~exist(fileparts(rand_model_file), 'file'),
				mkdir(fileparts(rand_model_file));
			end
			save( rand_model_file, 'svm' );	
			
		end
	end
	
	quit;
end
