function lsvrc2010_build_imdb()
	lsvrc2010_build_imdb_('train');
end

function lsvrc2010_build_imdb_(szPat)
	
	output_dir = '/net/per610a/export/das11f/plsang/LSVRC2010/metadata';
	output_file = sprintf('%s/imdb_%s.mat', output_dir, szPat);
	if exist(output_file, 'file'),
		fprintf('File [%s] already exist!', output_file);
		return;
	end
	
	root_kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image';
    kf_dir = sprintf('%s/%s', root_kf_dir, szPat);
	
	train_classes = dir (fullfile(kf_dir, 'n*'));
	
	imdb = struct;
	maxidx = 0;
	
	for ii = 1:length(train_classes),
        class_name = train_classes(ii).name;              
		
		fprintf(' [%d/%d] Building class [%s]...\n', ii, length(train_classes), class_name);
        class_kf_dir = fullfile(kf_dir, class_name);
		
		kfs = dir([class_kf_dir, '/*.JPEG']);
		
		imdb.(class_name).images = {};
		imdb.(class_name).indxs = [];	%universal index
		imdb.(class_name).uindxs = [];
		imdb.(class_name).paths = {};
		
		for jj = 1:length(kfs),
		
			img_name = kfs(jj).name;
			
			img_path = fullfile(class_kf_dir, img_name);
			
			imdb.(class_name).images{end+1} = img_name;
			
			imdb.(class_name).paths{end+1} = img_path;
			
			imdb.(class_name).indxs = maxidx + [1:length(kfs)];
			
			maxidx = maxidx + length(kfs);
			
		end
	end
	
	save(output_file, 'imdb');
	
end