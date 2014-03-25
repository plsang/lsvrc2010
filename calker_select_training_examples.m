%% Select randomly M classes, N images
function imdb = calker_select_training_examples(M, N)
	kf_dir = '/net/per610a/export/das09f/satoh-lab/dungmt/DataSet/LSVRC/2010/image/train';
	train_classes = dir (fullfile(kf_dir, 'n*'));
	
	selected_class_idxs = [1:length(train_classes)];
	class_rand_idx = randperm(length(train_classes));
	selected_class_idxs = selected_class_idxs(class_rand_idx(1:M));
	
	imdb = struct;
	
	for ii = selected_class_idxs,
		class_name = train_classes(ii).name;
		class_kf_dir = fullfile(kf_dir, class_name);
		
		kfs = dir([class_kf_dir, '/*.JPEG']);
        
		selected_img_idxs = [1:length(kfs)];	
		img_rand_idx = randperm(length(kfs));
		selected_img_idxs = selected_img_idxs(img_rand_idx(1:N));
		
		imdb.(class_name) = selected_img_idxs;
	end
end