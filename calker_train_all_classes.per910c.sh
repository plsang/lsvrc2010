matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 1, 'E', 333)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 334, 'E', 666)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'fea', 'covdet.hessian.sift.cb256.pca80.fisher', 'S', 667, 'E', 1000)" &
wait
