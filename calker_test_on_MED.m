%testset: KINDREDTEST (14464 clips), MEDTEST (27033 clips)
function calker_test_on_MED(M, N, R)
	
	set_env;
		
	%% dataset
	fprintf('Loading metadata...\n');
	medmd_file = sprintf('/net/per610a/export/das11f/plsang/trecvidmed13/metadata/common/metadata_devel.mat');
	metadata = load(medmd_file, 'metadata');
	metadata = metadata.metadata;
	
	clips = fieldnames(metadata);
	clear metadata;
	
	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	if ~exist(imdb_file, 'file'),
		error();
	end
	
	fprintf('Loading selected image db...\n');
	imdb = load(imdb_file, 'imdb');
	imdb = imdb.imdb;
	
	selected_classes = fieldnames(imdb);
	
	kf_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed13/feature/keyframes/covdet.hessian.sift.cb256.devel.fisher.pca/devel';
	
	fprintf('Loading all models...\n');
	
	models = struct;
	
	for kk = 1:M,
		
		class_name = selected_classes{kk};
		
		model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_rand%dc_%di/r%d/models/%s.mat', M, N, R, class_name);
		
		if ~exist(model_file, 'file'),
			error();
		end
		
		svm = load(model_file, 'svm'); 
		svm = svm.svm;
		
		models.(class_name).svind = svm.svind;
		models.(class_name).alphay = svm.alphay;
		models.(class_name).b = svm.b;
		models.(class_name).libsvm_cl = svm.libsvm_cl;
		
	end
	
	output_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/feature/covdet.hessian.sift.cb256.pca80.fisher';
	output_dir = sprintf('%s.att.M%d.N%d.R%d/devel', output_dir, M, N, R);
	
	if ~exist(output_dir, 'file'),
		mkdir(output_dir);
	end
	
	for ii=1:length(clips),
		fprintf('%d/%d clips processed...\n', ii, length(clips));
		clip_name = clips{ii};
		clip_att_fea_file = sprintf('%s/%s.mat', output_dir, clip_name);
		if exist(clip_att_fea_file, 'file'),
			continue;
		end
		
		clip_fea_file = sprintf('%s/%s.mat', kf_fea_dir, clip_name);
		if ~exist(clip_fea_file, 'file'),
			warning('File [%s] does not exist..\n', clip_fea_file);
			continue;
		end
		
		codes = load(clip_fea_file, 'codes');	
		codes = cat(2, codes.codes{:});
		
		%% removing NaN entries
		codes(:, any(isnan(codes), 1)) = [];
		
		if size(codes, 2) == 0,
			warning('Empty features after removing NaN [video: %s]\n', clip_name);
			continue;
		end
		
		% L2-norm
		codes = num2cell(codes, 1); %% convert to cell, group by column-wise operator
		codes = cellfun(@(x) x/norm(x, 2), codes, 'UniformOutput', false); %% apply l2-norm on each cell column
		codes = cell2mat(codes);	%% convert to matrix
			
		[N, Nt] = size(codes) ;
		
		scores = zeros(M, size(codes, 2)); % M x Nt
		
		
		for jj = 1:M,
			class_name = selected_classes{jj};
			
			test_codes = codes(models.(class_name).svind,:);
			
			scores(jj, :) = models.(class_name).alphay' * test_codes + models.(class_name).b;
			
			% using '-b 1' option to obtain propabilictic score
			% [y, acc, dec] = svmpredict(zeros(Nt, 1), [(1:Nt)' double(codes')], models.(class_name).libsvm_cl, '-b 1') ;		
			% scores = dec(:, 1)';
			
		end
		
		code = mean(scores, 2);
		save(clip_att_fea_file, 'code');
		
	end
	
end	