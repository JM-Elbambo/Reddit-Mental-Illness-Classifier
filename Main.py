from Classifier import Model
from DatasetCleaner import DatasetCleaner

if __name__ == "__main__":
	# File paths
	path_raw_training = r"Data Sets\Raw\Training Set.csv"
	path_raw_validation = r"Data Sets\Raw\Validation Set.csv"
	path_raw_test = r"Data Sets\Raw\Test Set.csv"
	path_processed_training = r"Data Sets\Processed\Training Set.csv"
	path_processed_validation = r"Data Sets\Processed\Validation Set.csv"
	path_processed_test = r"Data Sets\Processed\Test Set.csv"

	# Data cleaning
	print("\n============================================================\n")
	print("Cleaning:", path_raw_training)
	DatasetCleaner.clean_csv(path_raw_training, path_processed_training, ['title', 'post', 'class_id'])
	print("Cleaning:", path_raw_validation)
	DatasetCleaner.clean_csv(path_raw_validation, path_processed_validation, ['title', 'post', 'class_id'])
	print("Cleaning:", path_raw_test)
	DatasetCleaner.clean_csv(path_raw_test, path_processed_test, ['title', 'post', 'class_id'])

	# Train model
	# print("\n============================================================\n")
	# print("TRAINING PHASE")
	# model = Model(["title", "post"], "class_id", ["ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"])
	# model.train(path_processed_training)

	# print("\n============================================================\n")
	# print("HYPERPARAMETER TUNING")
	# train_grid_search = model.perform_grid_search(path_processed_training)
	# validation_grid_search = model.perform_grid_search(path_processed_validation)
	# print("Train: " + str(train_grid_search))
	# print("Validation: " + str(validation_grid_search))
	# model.graph_hyperparameter_tuning(path_processed_training, path_processed_validation)

	# Test model
	# print("\n============================================================\n")
	# print("TESTING PHASE")
	# model.test(path_processed_test)