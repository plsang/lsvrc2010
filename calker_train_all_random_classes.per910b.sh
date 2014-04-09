matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 1, 'E', 1000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 1001, 'E', 2000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 2001, 'E', 3000)" &
wait
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 3001, 'E', 4000)" &
matlab -nodisplay -r "calker_train_all_random_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 4001, 'E', 5000)" &
wait
