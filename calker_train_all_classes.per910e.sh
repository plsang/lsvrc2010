matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'S', 1, 'E', 250)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'S', 251, 'E', 500)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'S', 501, 'E', 750)" &
matlab -nodisplay -r "calker_train_all_classes(1000, 100, 1, 'S', 751, 'E', 1000)" &
wait
