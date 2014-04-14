
%% M classes, N images, R: random number
function calker_generate_random_classes(M, N, R, NumConcept)
	
	set_env;
	
	MaxPos = 100;
	MaxNeg = 10000; % max negative
	fea_pat = 'covdet.hessian.sift.cb256.pca80.fisher';	
	
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
	fprintf('Start generating...\n');
	
	
	%PosSet = [10, 100, 200, 300, 400, 500, 1000];
	PosSet = [10];
	
	for NumPos = PosSet,
		label_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/labels_C%05d_P%05d_N%05d.mat', M, N, R, NumConcept, NumPos, MaxNeg);
		if exist(label_file, 'file'),
			fprintf('File [%s] already exist \n', label_file);
		end
		
		labels = struct;

		%% seed for truly random with different matlab sessions, 
		%% otherwise everytime starting matlab, results will be the same
		stream = RandStream('mt19937ar', 'Seed', sum(100*clock));
		RandStream.setDefaultStream(stream);
		
		max_label = length(find(nonzero_idx>0));
		
		fprintf('--[%d/%d] Generating for class Pos = %d, Neg = %d...\n', find(PosSet == NumPos), length(PosSet), NumPos, MaxNeg);
				
		for kk = 1:NumConcept,
			class_name = sprintf('random_class_%05d', kk);
			
			%fprintf('[%d/%d] Generating for class ''%s''...\n', kk, NumConcept, class_name);	
			
			r_pos_idx = randperm(max_label);
			r_pos_idx = r_pos_idx(1:NumPos);
			
			r_neg_idx = setdiff(1:max_label, r_pos_idx);
			ridx = randperm(length(r_neg_idx));
			r_neg_idx = r_neg_idx(ridx(1:MaxNeg));
			
			r_train_idx = [r_pos_idx, r_neg_idx];
			
			labels.(class_name).pos_idx = r_pos_idx;
			labels.(class_name).neg_idx = r_neg_idx;
		end
		
		fprintf('Saving label file...\n');
		save( label_file, 'labels' );	
	end
	
	%NegSet = [10, 100, 200, 300, 400, 500, 1000];
	NegSet = [];
	
	for NumNeg = NegSet,
		label_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/labels_C%05d_P%05d_N%05d.mat', M, N, R, NumConcept, MaxPos, NumNeg);
		if exist(label_file, 'file'),
			fprintf('File [%s] already exist \n', label_file);
		end
		
		labels = struct;

		%% seed for truly random with different matlab sessions, 
		%% otherwise everytime starting matlab, results will be the same
		stream = RandStream('mt19937ar', 'Seed', sum(100*clock));
		RandStream.setDefaultStream(stream);
		
		max_label = length(find(nonzero_idx>0));
		
		fprintf('-- [%d/%d] Generating for class Pos = %d, Neg = %d...\n', find(NegSet == NumNeg), length(PosSet), MaxPos, NumNeg);	
		
		for kk = 1:NumConcept,
			class_name = sprintf('random_class_%05d', kk);
			
			r_pos_idx = randperm(max_label);
			r_pos_idx = r_pos_idx(1:MaxPos);
			
			r_neg_idx = setdiff(1:max_label, r_pos_idx);
			ridx = randperm(length(r_neg_idx));
			r_neg_idx = r_neg_idx(ridx(1:NumNeg));
			
			r_train_idx = [r_pos_idx, r_neg_idx];
			
			labels.(class_name).pos_idx = r_pos_idx;
			labels.(class_name).neg_idx = r_neg_idx;
		end
		
		fprintf('Saving label file...\n');
		save( label_file, 'labels' );	
	end
	
	
end

%% M classes, N images, R: random number
function calker_generate_random_classes_(M, N, R, NumConcept, MaxPos, MaxNeg)
	
	set_env;
	
	MaxPos = 100;
	MaxNeg = 10000; % max negative
	fea_pat = 'covdet.hessian.sift.cb256.pca80.fisher';	

	label_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/labels_P%05d.mat', M, N, R, MaxPos);
	if exist(label_file, 'file'),
		fprintf('File [%s] already exist \n', label_file);
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
	fprintf('Start generating...\n');
	
	labels = struct;

	%% seed for truly random with different matlab sessions, 
	%% otherwise everytime starting matlab, results will be the same
	stream = RandStream('mt19937ar', 'Seed', sum(100*clock));
	RandStream.setDefaultStream(stream);
	
	max_label = length(find(nonzero_idx>0));
	
	for kk = 1:NumConcept,
		class_name = sprintf('random_class_%05d', kk);
		
		fprintf('[%d/%d] Generating for class ''%s''...\n', kk, NumConcept, class_name);	
		
		r_pos_idx = randperm(max_label);
		r_pos_idx = r_pos_idx(1:MaxPos);
		
		r_neg_idx = setdiff(1:max_label, r_pos_idx);
		ridx = randperm(length(r_neg_idx));
		r_neg_idx = r_neg_idx(ridx(1:MaxNeg));
		
		r_train_idx = [r_pos_idx, r_neg_idx];
		
		labels.(class_name).pos_idx = r_pos_idx;
		labels.(class_name).neg_idx = r_neg_idx;
	end
	
	fprintf('Saving label file...\n');
	save( label_file, 'labels' );	
end
