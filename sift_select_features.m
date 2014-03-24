function sift_select_features( sift_algo, param )
%SELECT_FEATURES Summary of this function goes here
%   Detailed explanation goes here
	% nSize: step for dense sift
    % parameters
	
	%%

    max_features = 1000000;
	num_sample_kf_per_class = 100;	% number of sample keyframe / class
	
    kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image/train';
	train_classes = dir (fullfile(kf_dir, 'n*'));
	max_features_per_class = ceil(1.05*max_features/length(train_classes));
    
    
    feats = cell(length(train_classes), 1);

	output_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/feature/bow.codebook.devel/%s.%s.sift/data/selected_feats_%d.mat', sift_algo, num2str(param), max_features);
	if exist(output_file),
		fprintf('File [%s] already exist. Skipped\n', output_file);
		return;
	end
	
    parfor ii = 1:length(train_classes),
        class_name = train_classes(ii).name;
        
		class_kf_dir = fullfile(kf_dir, class_name);
		
		kfs = dir([class_kf_dir, '/*.JPEG']);
        
		selected_idx = [1:length(kfs)];
		if length(kfs) > num_sample_kf_per_class,
			rand_idx = randperm(length(kfs));
			selected_idx = selected_idx(rand_idx(1:num_sample_kf_per_class));
		end
		
		fprintf('Computing features for class: %d - %s %f %% complete\n', ii, class_name, ii/length(train_classes)*100.00);
		feat = [];
		for jj = selected_idx,
			img_name = kfs(jj).name;
			img_path = fullfile(class_kf_dir, img_name);
			im = imread(img_path);
			
			[frames, descrs] = sift_extract_features( im, sift_algo, param );
            
            % if more than 50% of points are empty --> possibley empty image
            if sum(all(descrs == 0, 1)) > 0.5*size(descrs, 2),
                warning('Maybe blank image...[%s]. Skipped!\n', img_name);
                continue;
            end
			feat = [feat descrs];
		end
        
        if size(feat, 2) > max_features_per_class,
            feats{ii} = vl_colsubset(feat, max_features_per_class);
        else
            feats{ii} = feat;
        end
        
    end
    
    % concatenate features into a single matrix
    feats = cat(2, feats{:});
    
    if size(feats, 2) > max_features,
         feats = vl_colsubset(feats, max_features);
    end

	output_dir = fileparts(output_file);
	if ~exist(output_dir, 'file'),
		%cmd = sprintf('mkdir -p %s', output_dir);
		%system(cmd);
		mkdir(output_dir);
	end
	
	fprintf('Saving selected features to [%s]...\n', output_file);
    save(output_file, 'feats', '-v7.3');
    
end

