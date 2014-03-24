function sift_main(sift_algo, param)

%% set environment variables
set_env;

if matlabpool('size') < 1,
	matlabpool open;
end	
sift_select_features(sift_algo, param); 
matlabpool close; 

proj_dir = '/net/per610a/export/das11f/plsang/LSVRC2010';
feat_pat = sprintf('%s.%s.sift', sift_algo, param);
dimred 	 = 80;

do_clustering_gmm(proj_dir, feat_pat, dimred); 


end