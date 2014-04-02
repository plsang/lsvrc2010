
%% M classes, N images, R: random number
function calker_cal_all_kernels(M, N, R, varargin)
	
	szPat = 'train';
	ker_root_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/kernels/train';
	ker_dir = sprintf('%s/lsvrc2010_rand%dc_%di/r%d', ker_root_dir, M, N, R);
	
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
	
	codes = cell(length(train_classes));
	
	parfor kk = 1:length(train_classes),
		class_name_ii = train_classes{kk};
		selected_img_idx_ii = imdb.(class_name_ii);
		feat_file_ii = fullfile(fea_dir, [class_name_ii, '.mat']);
		fprintf('\n [%d/%d] loading feature file...\n', kk, length(train_classes));
		
		codes_kk = load(feat_file_ii, 'codes');
		codes_kk = codes_kk.codes(selected_img_idx_ii);
		codes_kk = cat(2, codes_kk{:});
		
		%% whitening NaN entries
		codes_kk(:, any(isnan(codes_kk), 1)) = 0;
		
		codes_kk = num2cell(codes_kk, 1); %% convert to cell, group by column-wise operator
		codes_kk = cellfun(@(x) x/norm(x, 2), codes_kk, 'UniformOutput', false); %% apply l2-norm on each cell column
		codes_kk = cell2mat(codes_kk);	%% convert to matrix
		
		codes{kk} = codes_kk;
	end
	
	for ii = start_class:end_class,
	
		class_name_ii = train_classes{ii};
		selected_img_idx_ii = imdb.(class_name_ii);
		
		ker_file = sprintf('%s/%s.mat', ker_dir, class_name_ii);
		
		if exist(ker_file, 'file'),
			fprintf('Skipped calculating kernel %s \n', class_name_ii);
			continue;
		end
		
		codes_ii = codes{ii};
		
		kers = cell(length(train_classes), 1);
		
		
		parfor jj = 1:length(train_classes),
			codes_jj = codes{jj};
			
			kers{jj} = codes_ii' * codes_jj;
					
		end
		
		fprintf('[%d/%d] Concatenating kernels into a single matrix...', ii - start_class + 1, end_class - start_class + 1);
		kers = cat(2, kers{:});
		save( ker_file, 'kers', '-v7.3');
		%save( ker_file, 'kers');
		
	end

end

function par_save( output_file, ker )
	save( output_file, 'ker');
end
