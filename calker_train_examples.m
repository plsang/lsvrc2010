
%% M classes, N images, R: random number
function calker_train_examples(M, N, R, start_class, end_class)
	
	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	if exist(imdb_file, 'file'),
		imdb = load(imdb_file, 'imdb');
		imdb = imdb.imdb;
	else
		imdb = calker_select_training_examples(M, N);
		if ~exist(fileparts(imdb_file), 'file'),
			mkdir(fileparts(imdb_file));
		end
		save(imdb_file, 'imdb');
	end
	
	%% loading features
	fea_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/feature/covdet.hessian.sift.cb256.pca80.fisher/train';
	
	selected_classes = fieldnames(imdb);
	
	hists_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/hists.mat', M, N, R);
	labels_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/labels.mat', M, N, R);
	train_ker_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/train_ker.mat', M, N, R);
	
	if exist(hists_file, 'file') && exist(labels_file, 'file'),
		hists = load(hists_file, 'hists');
		labels = load(labels_file, 'labels');
		
		hists = hists.hists;
		labels = labels.labels;
	else
		hists = cell(length(selected_classes), 1);	
		labels = cell(length(selected_classes), 1);
		
		for ii = 1:length(selected_classes),
			class_name = selected_classes{ii};
			selected_img_idx = imdb.(class_name);
			
			feat_file = fullfile(fea_dir, [class_name, '.mat']);
			fprintf(' [%d/%d] loading feature file...\n', ii, length(selected_classes));
			codes = load(feat_file, 'codes');
			
			selected_img_feats = codes.codes(selected_img_idx);
			selected_img_feats = cat(2, selected_img_feats{:});
			
			%% removing NaN entries
			selected_img_feats(:, any(isnan(selected_img_feats), 1)) = [];
			
			hists{ii} = selected_img_feats; 
			labels{ii} = ii*ones(size(selected_img_feats, 2), 1);
		end
		
		hists = cat(2, hists{:});
		labels = cat(1, labels{:});
		
		fprintf('Saving hists and labels...\n');
		if ~exist(fileparts(hists_file), 'file'),
			mkdir(fileparts(hists_file));
		end
		save(hists_file, 'hists', '-v7.3');
		save(labels_file, 'labels');
		
	end
	
	if ~exist(train_ker_file, 'file'),	
		fprintf('Calculating training ker using l2 distances...\n');
		train_ker = hists' * hists ;
		
		fprintf('Saving training ker...\n');
		save(train_ker_file, 'train_ker', '-v7.3');
	else
		train_ker = load(train_ker_file, 'train_ker');
		train_ker = train_ker.train_ker;
	end
	
	labels = double(labels);
	train_ker = double(train_ker);
	
	fprintf('preparing labels...\n');
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
	
	if ~exist('start_class', 'var') || start_class < 1,
        start_class = 1;
    end
    
    if ~exist('end_class', 'var') || end_class > M,
        end_class = M;
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
						   'crossvalidation', 5, ...
						   'weights', [+1 posWeight ; -1 1]') ;
		
		svm = svmflip(svm, labels);	
		
		if 1, % test_on_train
			
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