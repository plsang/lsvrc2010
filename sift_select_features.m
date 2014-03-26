function sift_select_features( proj_name, sift_algo, param )
%SELECT_FEATURES Summary of this function goes here
%   Detailed explanation goes here
	% nSize: step for dense sift
    % parameters
	
	%%

    max_features = 1000000;
    sample_length = 5; % frames
    
    segments = load_segments(proj_name, 'devel', 'keyframe-100000');
    max_features_per_video = ceil(1.05*max_features/length(segments));
    
    
    feats = cell(length(segments), 1);

	output_file = sprintf('/net/per900a/raid0/plsang/%s/feature/bow.codebook.%s.devel/%s.%s.sift/data/selected_feats_%d.mat', proj_name, proj_name, sift_algo, num2str(param), max_features);
	if exist(output_file),
		fprintf('File [%s] already exist. Skipped\n', output_file);
		return;
	end
    
	kf_dir = sprintf('/net/per900a/raid0/plsang/%s/keyframes/devel', proj_name);
	
	if strcmp(proj_name, 'trecvidmed10'),
		kf_dir = sprintf('/net/per900a/raid0/plsang/%s/keyframes', proj_name);
	end
	
    parfor ii = 1:length(segments),
        segment = segments{ii};
        %segment = 'HVC5295.shot006.frame9001_10800';
        pattern =  '(?<video>\w+)\.\w+\.frame(?<start>\d+)_(?<end>\d+)';
        info = regexp(segment, pattern, 'names');
        
        %kf_dir = '/net/per900a/raid0/plsang/trecvidmed11/keyframes/devel';
        video_kf_dir = fullfile(kf_dir, info.video);
        
		kfs = dir([video_kf_dir, '/*.jpg']);
        
		selected_idx = [1:length(kfs)];
		if length(kfs) > sample_length,
			rand_idx = randperm(length(kfs));
			selected_idx = selected_idx(rand_idx(1:sample_length));
		end
		
		fprintf('Computing features for: %d - %s %f %% complete\n', ii, segment, ii/length(segments)*100.00);
		feat = [];
		for jj = selected_idx,
			img_name = kfs(jj).name;
			img_path = fullfile(video_kf_dir, img_name);
			im = imread(img_path);
			
			[frames, descrs] = sift_extract_features( im, sift_algo, param )
            
            % if more than 50% of points are empty --> possibley empty image
            if sum(all(descrs == 0, 1)) > 0.5*size(descrs, 2),
                warning('Maybe blank image...[%s]. Skipped!\n', img_name);
                continue;
            end
			feat = [feat descrs];
		end
        
        if size(feat, 2) > max_features_per_video,
            feats{ii} = vl_colsubset(feat, max_features_per_video);
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
		cmd = sprintf('mkdir -p %s', output_dir);
		system(cmd);
	end
	
	fprintf('Saving selected features to [%s]...\n', output_file);
    save(output_file, 'feats', '-v7.3');
    
end

