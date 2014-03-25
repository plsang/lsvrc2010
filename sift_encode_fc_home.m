function [ output_args ] = sift_encode_fc_home( proj_dir, szPat, sift_algo, param, dimred, codebook_size, spm, start_class, end_class)

	% update: Jun 25th, SPM suported
    % setting
    set_env;
        
    % encoding type
    enc_type = 'fisher';
	
	if ~exist('codebook_size', 'var'),
		codebook_size = 256;
	end
    
	if ~exist('spm', 'var'),
		spm = 0;
	end
	
	default_dim = 128;
	if ~exist('dimred', 'var'),
		dimred = default_dim;
	end
	
	feature_ext = sprintf('%s.%s.sift.cb%d', sift_algo, num2str(param), codebook_size);
	if spm > 0,
		feature_ext = sprintf('%s.spm', feature_ext);
	end
	
	if dimred < default_dim,,
		feature_ext = sprintf('%s.pca%d', feature_ext, dimred);
	end
	
    output_dir = sprintf('%s/feature/%s.%s/%s', proj_dir, feature_ext, enc_type, szPat) ;
    if ~exist(output_dir, 'file'),
		mkdir(output_dir);
	end
    
    codebook_file = sprintf('%s/feature/bow.codebook.devel/%s.%s.sift/data/codebook.gmm.%d.%d.mat', ...
		proj_dir, sift_algo, num2str(param), codebook_size, dimred);
		
	fprintf('Loading codebook [%s]...\n', codebook_file);
    codebook_ = load(codebook_file, 'codebook');
    codebook = codebook_.codebook;
 
	
 	low_proj = [];
	if dimred < default_dim,
		lowproj_file = sprintf('%s/feature/bow.codebook.devel/%s.%s.sift/data/lowproj.%d.%d.mat', ...
			proj_dir, sift_algo, num2str(param), dimred, default_dim);
			
		fprintf('Loading low projection matrix [%s]...\n', lowproj_file);
		low_proj_ = load(lowproj_file, 'low_proj');
		low_proj = low_proj_.low_proj;
	end
	
	root_kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image';
    kf_dir = sprintf('%s/%s', root_kf_dir, szPat);
    
	train_classes = dir (fullfile(kf_dir, 'n*'));
	
	if ~exist('start_class', 'var') || start_class < 1,
        start_class = 1;
    end
    
    if ~exist('end_class', 'var') || end_class > length(train_classes),
        end_class = length(videos);
    end
	
    for ii = start_class:end_class,
        class_name = train_classes(ii).name;              
    
        class_kf_dir = fullfile(kf_dir, class_name);
		
		kfs = dir([class_kf_dir, '/*.JPEG']);
        
		output_file = [output_dir, '/', class_name, '.mat'];
		if exist(output_file, 'file'),
			fprintf('File [%s] already exist. Skipped!!\n', output_file);
			continue;
		end
		
		codes = cell(length(kfs), 1);
		
		fprintf(' [%d/%d] Extracting and Encoding for image class [%s]...\n', ii, length(train_classes), class_name);
		
		parfor jj = 1:length(kfs),
			if ~mod(jj, 50),
				fprintf('%d ', jj);
			end
			
			img_name = kfs(jj).name;
			
			img_path = fullfile(class_kf_dir, img_name);
			im = imread(img_path);
			
			[frames, descrs] = sift_extract_features( im, sift_algo, param );
			
			% if more than 50% of points are empty --> possibley empty image
            if sum(all(descrs == 0, 1)) > 0.5*size(descrs, 2),
                warning('Maybe blank image...[%s]. Skipped!\n', img_name);
                continue;
            end
			
			if spm > 0
				code = sift_encode_spm(enc_type, size(im), frames, descrs, codebook, [], low_proj);
			else
				code = sift_do_encoding(enc_type, descrs, codebook, [], low_proj);	
			end
			
			codes{jj} = code;
		end
		
        par_save(output_file, codes); % MATLAB don't allow to save inside parfor loop             
        
    end
    
    %toc
    % quit;

end

function par_save( output_file, codes )
  save( output_file, 'codes');
end

function log (msg)
	fh = fopen('sift_encode_fc_home.log', 'a+');
    msg = [msg, ' at ', datestr(now), '\n'];
	fprintf(fh, msg);
	fclose(fh);
end

