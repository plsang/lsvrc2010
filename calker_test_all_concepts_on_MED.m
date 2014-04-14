%testset: KINDREDTEST (14464 clips), MEDTEST (27033 clips)
% 48396 
function calker_test_all_concepts_on_MED(M, N, R, C, Pos, Neg, varargin)
	
	%% C: num concept
	set_env;
	
	start_class = 1;
	end_class = C;
	
	fea_pat = 'covdet.hessian.sift.cb256.pca80.fisher';
	
	for k=1:2:length(varargin),	
		opt = lower(varargin{k});
		arg = varargin{k+1} ;
		switch opt
			case 's'
				start_class = arg ;
			case 'e' ;
				end_class = arg ;
			case 'fea'
				fea_pat = arg;	
			otherwise
				error(sprintf('Option ''%s'' unknown.', opt)) ;
		end  
	end
	
	%% dataset
	fprintf('Loading metadata...\n');
	
	medmd_file = '/net/per610a/export/das11f/plsang/trecvidmed13/metadata/medmd.mat';
	load(medmd_file, 'MEDMD'); 
	
	clips = [MEDMD.EventKit.EK130Ex.clips, MEDMD.EventBG.default.clips, MEDMD.RefTest.KINDREDTEST.clips, MEDMD.RefTest.MEDTEST.clips];
	clips = unique(clips);	% 48396 clips
	
	clear MEDMD;
	
	kf_fea_dir = '/net/per610a/export/das11f/plsang/trecvidmed13/feature/keyframes/covdet.hessian.sift.cb256.devel.fisher.pca/devel';
	
	fprintf('Loading all models...\n');
	
	models = struct;
	
	for kk = 1:C,
		
		class_name = sprintf('random_class_%05d', kk);
		
		model_file = sprintf('/net/per610a/export/das11f/plsang/LSVRC2010/experiments/lsvrc2010_M%d_N%d_R%d/%s/random_models_C%05d_P%05d_N%05d/%s.mat', M, N, R, fea_pat, C, Pos, Neg, class_name);
		
		if ~exist(model_file, 'file'),
			error('File not found [%s]\n', model_file);
		end
		
		svm = load(model_file, 'svm'); 
		svm = svm.svm;
		
		models.(class_name).svind = svm.svind;
		models.(class_name).alphay = svm.alphay;
		models.(class_name).b = svm.b;
		%models.(class_name).libsvm_cl = svm.libsvm_cl;
		
	end
	
	med_output_dir = '/net/per610a/export/das11f/plsang/trecvidmed13/feature/segment-att';
	output_dir = sprintf('%s/%s', med_output_dir, fea_pat);
	
	output_dir = sprintf('%s.att.M%d.N%d.R%d.C%d.P%05d.N%05d/devel', output_dir, M, N, R, C, Pos, Neg);
	
	if ~exist(output_dir, 'file'),
		mkdir(output_dir);
	end
	
	for ii=start_class:end_class,
		fprintf('%d/%d clips processed...\n', ii-start_class + 1, end_class - start_class + 1);
		clip_name = clips{ii};
		clip_att_fea_file = sprintf('%s/%s.mat', output_dir, clip_name);
		if exist(clip_att_fea_file, 'file'),
			continue;
		end
		
		clip_fea_file = sprintf('%s/%s.mat', kf_fea_dir, clip_name);
		if ~exist(clip_fea_file, 'file'),
			warning('File [%s] does not exist..\n', clip_fea_file);
			continue;
		end
		
		codes = load(clip_fea_file, 'codes');	
		codes = cat(2, codes.codes{:});
		
		%% removing NaN entries
		codes(:, any(isnan(codes), 1)) = [];
		
		if size(codes, 2) == 0,
			warning('Empty features after removing NaN [video: %s]\n', clip_name);
			continue;
		end
		
		% L2-norm
		codes = num2cell(codes, 1); %% convert to cell, group by column-wise operator
		codes = cellfun(@(x) x/norm(x, 2), codes, 'UniformOutput', false); %% apply l2-norm on each cell column
		codes = cell2mat(codes);	%% convert to matrix
			
		[N, Nt] = size(codes) ;
		
		scores = zeros(C, size(codes, 2)); % C x Nt
		
		for jj = 1:C,
			class_name = sprintf('random_class_%05d', jj);
			
			test_codes = codes(models.(class_name).svind,:);
			
			scores(jj, :) = models.(class_name).alphay' * test_codes + models.(class_name).b;
			
		end
		
		code = mean(scores, 2);
		save(clip_att_fea_file, 'code');
		
	end
	
	quit;
	
end	
