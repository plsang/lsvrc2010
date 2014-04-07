function calker_main(M, N, R)
	calker_select_training_examples(M, N, R);
	calker_cal_all_kernels(M, N, R);
	calker_train_all_classes(M, N, R);
end