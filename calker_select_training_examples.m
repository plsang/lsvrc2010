%% Select randomly M classes, N images
function calker_select_training_examples(M, N, R)

	imdb_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/metadata/lsvrc2010_rand%dc_%di/r%d/imdb.mat', M, N, R);
	
	if exist(imdb_file, 'file'),
		fprintf('File already exist!');
		return;	
	end
	
	kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image/train';
	train_classes = dir (fullfile(kf_dir, 'n*'));
	
	selected_class_idxs = [1:length(train_classes)];
	class_rand_idx = randperm(length(train_classes));
	selected_class_idxs = selected_class_idxs(class_rand_idx(1:M));
	
	imdb = struct;
	
	count = 1;
	for ii = selected_class_idxs,
		if mod(count, 10) == 0, 
			fprintf('%d/%d \r', count, length(selected_class_idxs));
		end
		class_name = train_classes(ii).name;
		class_kf_dir = fullfile(kf_dir, class_name);
		
		kfs = dir([class_kf_dir, '/*.JPEG']);
        
		selected_img_idxs = [1:length(kfs)];	
		img_rand_idx = randperm(length(kfs));
		selected_img_idxs = selected_img_idxs(img_rand_idx(1:N));
		
		imdb.(class_name) = selected_img_idxs;
		count = count + 1;
	end
	
	if ~exist(fileparts(imdb_file), 'file'),
		mkdir(fileparts(imdb_file));
	end
	save(imdb_file, 'imdb');
end