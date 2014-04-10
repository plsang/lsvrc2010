matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 5001, 'E', 6000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 6001, 'E', 7000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 7001, 'E', 8000)" &
wait
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 8001, 'E', 9000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 9001, 'E', 10000)" &
wait
