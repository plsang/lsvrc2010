
%% M classes, N images, R: random number
function calker_train_all_classes_liblinear(M, N, R, varargin)
	
	set_env;
	
	szPat = 'train';
	ker_root_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/kernels/train';
	ker_dir = sprintf('%s/lsvrc2010_rand%dc_%di/r%d', ker_root_dir, M, N, R);
	feat_dim = 40960;
	
	if ~exist(ker_dir, 'file'),
		mkdir(ker_dir);
	end
	
	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	
	if ~exist(imdb_file, 'file'),
		error();
	end
	
	fprintf('Loading imdb ...\n');
	imdb = load(imdb_file, 'imdb');
	imdb = imdb.imdb;

	train_classes = fieldnames(imdb);
		
	%% loading features
	fea_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/feature/covdet.hessian.sift.cb256.pca80.fisher/train';
	
		
	root_kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image';
	kf_dir = sprintf('%s/%s', root_kf_dir, szPat);
	
	start_class = 1;
	end_class = length(train_classes);
	
	
	for k=1:2:length(varargin),
	
		opt = lower(varargin{k});
		arg = varargin{k+1} ;
	  
		switch opt
			case 'pat'
				szPat = arg ;
			case 's'
				start_class = arg ;
			case 'e' ;
				end_class = arg ;
			otherwise
				error(sprintf('Option ''%s'' unknown.', opt)) ;
		end  
	end
	
	fprintf('Initializing codes...\n');	
	codes = zeros(M*N, feat_dim);
	
	fprintf('Initializing labels...\n');
	all_labels = -ones(M*N, M);
	
	fprintf('Assembling pre-computed kernels...\n');
	label_idx = 0;
	for kk = 1:length(train_classes),
		class_name_ii = train_classes{kk};
		selected_img_idx_ii = imdb.(class_name_ii);
		feat_file_ii = fullfile(fea_dir, [class_name_ii, '.mat']);
		fprintf('\n [%d/%d] loading feature file...\n', kk, length(train_classes));
		
		codes_kk = load(feat_file_ii, 'codes');
		codes_kk = codes_kk.codes(selected_img_idx_ii);
		codes_kk = cat(2, codes_kk{:});
		
		%% Removing NaN entries
		codes_kk(:, any(isnan(codes_kk), 1)) = 0;
		
		codes_kk = num2cell(codes_kk, 1); %% convert to cell, group by column-wise operator
		codes_kk = cellfun(@(x) x/norm(x, 2), codes_kk, 'UniformOutput', false); %% apply l2-norm on each cell column
		codes_kk = cell2mat(codes_kk);	%% convert to matrix
		
		img_idx = label_idx + [1:N]; 
		
		codes(img_idx, :) = codes_kk';
		
		all_labels(img_idx, kk) = 1;
		
		label_idx = label_idx + N;
	end
	
	fprintf('Removing all-zero indexes...\n');
	nonzero_idx = any(codes, 2);
	codes = codes(nonzero_idx, :); 
	all_labels = all_labels(nonzero_idx, :);
	
	fprintf('Converting to sparse features...');
	tic; codes = sparse(codes); toc;
	
	for kk = start_class:end_class,
	
		class_name = train_classes{kk};
		
		model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/linear-models/%s.mat', M, N, R, class_name);
		
		if exist(model_file, 'file'),
			fprintf('Skipped training %s \n', model_file);
			continue;
		end
		
		labels = double(all_labels(:, kk));
		
		fprintf('[%d/%d] Training class ''%s''...\n', kk - start_class + 1, end_class - start_class + 1, class_name);	
				
		tic; svm = train(labels, codes); toc;
		
		fprintf('\tSaving model ''%s''.\n', model_file) ;
		if ~exist(fileparts(model_file), 'file'),
			mkdir(fileparts(model_file));
		end
		
		save( model_file, 'svm' );	
		
	end

end

function par_save( output_file, ker )
	save( output_file, 'ker');
end
