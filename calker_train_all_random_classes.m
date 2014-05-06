
%% M classes, N images, R: random number
function calker_train_all_random_classes(M, N, R, varargin)
	
	set_env;
	
	start_class = 1;
	end_class = M;
	cv = 0;	% cross validation
	C = 1;	% C parameter for SVM
	
	fea_pat = 'covdet.hessian.sift.cb256.pca80.fisher';
	
	Pos = 100;
	Neg = 10000;
	num_concept = 10000;
	
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
			case 'pos' ;
				Pos = arg ;
			case 'neg' ;
				Neg = arg ;
			case 'nc' ;
				num_concept = arg ;	
			case 'fea'
				fea_pat = arg;	
			otherwise
				error(sprintf('Option ''%s'' unknown.', opt)) ;
		end  
	end

	%imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_M%d_N%d_R%d/imdb.mat', M, N, R);
	if ~exist(imdb_file, 'file'),
		error();
	end
	
	fprintf('Loading selected image db...\n');
	imdb = load(imdb_file, 'imdb');
	imdb = imdb.imdb;
	
	selected_classes = fieldnames(imdb);	
	
	ker_root_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/kernels/train';
	%ker_dir = sprintf('%s/lsvrc2010_rand%dc_%di/%s/r%d', ker_root_dir, M, N, fea_pat, R);
	ker_dir = sprintf('%s/lsvrc2010_M%d_N%d_R%d/%s', ker_root_dir, M, N, R, fea_pat);
	
	fprintf('Initializing pre-computed kernels...\n');
	
	labels = ones(1, M*N);
	train_ker = zeros(M*N, M*N);
	
	fprintf('Initializing labels...\n');
	
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
		
		label_idx = label_idx + N;
	end
	
	fprintf('Removing all-zero indexes...\n');
	nonzero_idx = any(train_ker, 1);
	train_ker = train_ker(nonzero_idx, nonzero_idx); 
	labels = double(labels(nonzero_idx));
	
	%label_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/labels_C%05d_P%05d_N%05d.mat', M, N, R, num_concept, Pos, Neg);
	label_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_M%d_N%d_R%d/labels_C%05d_P%05d_N%05d.mat', M, N, R, NumConcept, NumPos, MaxNeg);
	if ~exist(label_file, 'file'),
		error();
	end
	fprintf('Loading labels...\n');
	rlabels = load(label_file, 'labels');
	rlabels = rlabels.labels;
	
	fprintf('Start training...\n');
	for kk = start_class:end_class,
		class_name = sprintf('random_class_%05d', kk);
		
		fprintf('[%d/%d] Training class ''%s''...\n', kk - start_class + 1, end_class - start_class + 1, class_name);	
		
		fprintf('Learning random unlabelled models...\n');
		
		rand_model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_M%d_N%d_R%d/%s/random_models_C%05d_P%05d_N%05d/%s.mat', ...
			M, N, R, fea_pat, num_concept, Pos, Neg, class_name);
		
		if exist(rand_model_file, 'file'),
			fprintf('Skipped training %s \n', rand_model_file);
			continue;
		end

		r_pos_idx = rlabels.(class_name).pos_idx;
		r_neg_idx = rlabels.(class_name).neg_idx;
		
		r_train_idx = [r_pos_idx, r_neg_idx];
		
		labels(r_pos_idx) = 1;
		labels(r_neg_idx) = -1;
		
		posWeight = ceil(length(r_neg_idx)/length(r_pos_idx));
		
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
