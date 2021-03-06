
%% M classes, N images, R: random number
function calker_cal_train_kernels(M, N, R)
	
	train_ker_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/train_ker.mat', M, N, R);
	
	if exist(train_ker_file, 'file'),
		fprintf('Training ker is already calculated...\n');
		return;
	end
	
	
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
			
			% important applying L2-norm before calculating distance matrix
			fprintf('Normalizing hists...\n');
			selected_img_feats = num2cell(selected_img_feats, 1); %% convert to cell, group by column-wise operator
			selected_img_feats = cellfun(@(x) x/norm(x, 2), selected_img_feats, 'UniformOutput', false); %% apply l2-norm on each cell column
			selected_img_feats = cell2mat(selected_img_feats);	%% convert to matrix
	
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
	
	fprintf('Calculating training ker using l2 distances...\n');
	train_ker = hists' * hists ;
	
	fprintf('Saving training ker...\n');
	save(train_ker_file, 'train_ker', '-v7.3');



end