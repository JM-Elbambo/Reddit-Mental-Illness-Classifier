from Classifier import Model
from DatasetCleaner import DatasetCleaner
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# File paths
	path_raw_training = r"Data Sets\Raw\Training Set.csv"
	path_raw_validation = r"Data Sets\Raw\Validation Set.csv"
	path_raw_test = r"Data Sets\Raw\Test Set.csv"
	path_processed_training = r"Data Sets\Processed\Training Set.csv"
	path_processed_validation = r"Data Sets\Processed\Validation Set.csv"
	path_processed_test = r"Data Sets\Processed\Test Set.csv"

	# Clean Data
	print("\n============================================================\n")
	print("Cleaning:", path_raw_training)
	DatasetCleaner.clean_csv(path_raw_training, path_processed_training, ['title', 'post', 'class_id'])
	print("Cleaning:", path_raw_validation)
	DatasetCleaner.clean_csv(path_raw_validation, path_processed_validation, ['title', 'post', 'class_id'])
	print("Cleaning:", path_raw_test)
	DatasetCleaner.clean_csv(path_raw_test, path_processed_test, ['title', 'post', 'class_id'])

	# Train model
	print("\n============================================================\n")
	print("TRAINING PHASE")
	model = Model(["title", "post"], "class_id", ["ADHD", "Anxiety", "Bipolar", "Depression", "PTSD", "None"])
	model.train(path_processed_training)

	# Perform grid search
	# print("\n============================================================\n")
	# print("GRID SEARCH")
	# train_grid_search = model.perform_grid_search(path_processed_training)
	# validation_grid_search = model.perform_grid_search(path_processed_validation)
	# print("Train: " + str(train_grid_search))
	# print("Validation: " + str(validation_grid_search))
	# model.graph_hyperparameter_tuning(path_processed_training, path_processed_validation)

	# Perform hyperparameter turning
	print("\n============================================================\n")
	print("HYPERPARAMETER TUNING")
	parameter_names = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes")
	parameter_names = ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_leaf_nodes")
	parameter_count = len(parameter_names)
	x_axes, train_scores, validation_scores = model.get_hyperparameter_tuning_scores(path_processed_training, path_processed_validation, 40, 10)
	# Plot scores
	rows = 2
	cols = 3
	fig, ax = plt.subplots(nrows=rows, ncols=cols) #figsize=(18, 5)
	for row in range(rows):
		for col in range(cols):
			index = row * cols + col
			if (index == len(x_axes)):
				break
			ax[row, col].plot(x_axes[index], train_scores[index], ".-", label="Train")
			ax[row, col].plot(x_axes[index], validation_scores[index], ".-", label="Validation")
			ax[row, col].set_xlabel("Value")
			ax[row, col].set_ylabel("F1 Score")
			ax[row, col].set_title(parameter_names[index])
			ax[row, col].legend(loc="best")
			ax[row, col].grid()
	plt.tight_layout()
	plt.show()

	# Test model
	print("\n============================================================\n")
	print("TESTING PHASE: TRAIN SET")
	model.test(path_processed_training)

	print("\n============================================================\n")
	print("TESTING PHASE: VALIDATION SET")
	model.test(path_processed_validation)

	print("\n============================================================\n")
	print("TESTING PHASE: TEST SET")
	model.test(path_processed_test)