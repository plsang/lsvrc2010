matlab -nodisplay -r "calker_train_all_classes(1000, 100, 8, 'S', 1, 'E', 333)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 8, 'S', 334, 'E', 666)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 8, 'S', 667, 'E', 1000)" &
wait
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 9, 'S', 1, 'E', 333)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 9, 'S', 334, 'E', 666)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 9, 'S', 667, 'E', 1000)" &
wait
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 10, 'S', 1, 'E', 333)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 10, 'S', 334, 'E', 666)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 10, 'S', 667, 'E', 1000)" &
wait
