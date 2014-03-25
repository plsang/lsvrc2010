function sift_main(sift_algo, param)

%% set environment variables
set_env;

proj_dir = '/net/per610a/export/das11f/plsang/LSVRC2010';
feat_pat = sprintf('%s.%s.sift', sift_algo, param);
dimred 	 = 80;

if matlabpool('size') < 1,
	matlabpool open 4;
end	
sift_select_features(sift_algo, param); 
matlabpool close; 

do_clustering_gmm(proj_dir, feat_pat, dimred); 

if matlabpool('size') < 1,
	matlabpool open 8;
end	
sift_encode_fc_home(proj_dir, 'train', 'covdet', 'hessian', 80);
matlabpool close; 

end