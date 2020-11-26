import Properties as pr
import Utils as u
import sys
import numpy as np
import scipy as sp
import pickle
import math
import time
from dependencies.prettytable import PrettyTable
import CustomPlot as shplot
from sklearn.cluster import DBSCAN
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def update_properties():
	if pr.real_data:
		pr.real_data_testing_size = pr.real_dataset_size - pr.real_data_training_size
		pr.number_of_users = math.ceil(pr.real_data_training_size / (1-pr.fraction_of_deviators))
		pr.number_of_test_users = math.ceil(pr.real_data_testing_size / (1-pr.fraction_of_deviators))

	pr.number_of_trustworthy_users = int((1-pr.fraction_of_deviators)*pr.number_of_users)
	pr.number_of_deviating_users = pr.number_of_users - pr.number_of_trustworthy_users

	pr.number_of_trustworthy_test_users = int((1-pr.fraction_of_deviators)*pr.number_of_test_users)
	pr.number_of_deviating_test_users = pr.number_of_test_users - pr.number_of_trustworthy_test_users


def generate_data(stage='Training'):
	if stage == 'Testing':
		if pr.real_data: read_start_index = pr.real_data_training_size
		if pr.real_data: read_end_index = pr.real_dataset_size
		normal_sample_count = pr.number_of_trustworthy_test_users
		deviating_sample_count = pr.number_of_deviating_test_users
	else:
		if pr.real_data: read_start_index = 0
		if pr.real_data: read_end_index = pr.real_data_training_size
		normal_sample_count = pr.number_of_trustworthy_users
		deviating_sample_count = pr.number_of_deviating_users

	# Generate normal samples from distribution
	if pr.real_data:
		u.load_files()
		normal_samples = u.read_samples('sensors', pr.sensor_index, read_start_index, read_end_index)
	else:
		normal_samples = u.generate_samples(pr.normal_mu, pr.normal_sigma, pr.number_of_measurements, normal_sample_count)

	# Generate deviating samples	
	if pr.adversary == 'normal':
		deviating_samples = u.generate_samples(pr.malicious_mu, pr.malicious_sigma, pr.number_of_measurements, deviating_sample_count)

	elif pr.adversary == 'uniform':
		deviating_samples = u.generate_samples_uniform(pr.number_of_measurements, deviating_sample_count)

	elif pr.adversary == 'gamma':
		deviating_samples = u.generate_samples_gamma(pr.malicious_mu, pr.malicious_sigma, pr.number_of_measurements, deviating_sample_count)

	elif pr.adversary == 'random':
		deviating_samples = u.generate_samples_random(pr.number_of_measurements, deviating_sample_count)

	samples = np.vstack((normal_samples, deviating_samples))
	true_labels = np.hstack((np.zeros(len(normal_samples), dtype=np.int), np.ones(len(deviating_samples), dtype=np.int)))
	
	return [samples, true_labels]


def calculate_metrics(expected_labels, found_labels):
	tn, fp, fn, tp = confusion_matrix(expected_labels, found_labels).ravel()
	acc = (tp + tn) / (tp + tn + fp + fn)
	if tp + fn != 0:
		recall = tp / (tp + fn)
	else: 
		recall = 0.0
	if tp + fp != 0:
		precision = tp / (tp + fp)
	else: 
		precision = 0.0
	
	return acc, recall, precision


def cluster_data(samples):
	# Calculate spatial distances for DBSCAN
	distances = sp.spatial.distance.pdist(samples, u.distance_ds_conflict)
	distances_sqform = sp.spatial.distance.squareform(distances)
	epsilon = np.percentile(distances, pr.epsilon)
	
	# Complete DBSCAN
	dbscan = DBSCAN(eps=epsilon, min_samples=pr.fraction_of_deviators * pr.number_of_users, metric='precomputed').fit(distances_sqform)

	# Flag smallest cluster
	u.largest_cluster(dbscan.labels_)
	
	return np.absolute(dbscan.labels_)


def classify(training_samples, test_samples, test_labels, cluster_gen_labels, debug):
	# Try Neural Network
	debug.write("Training Neural Network..\n")
	train_samples_shuffled, cluster_gen_labels_shuffled = unison_shuffled_copies(training_samples, cluster_gen_labels)
	mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(50, 20), shuffle=True, learning_rate_init=0.02, tol=0.0000000000001, solver='adam', max_iter=10000)	
	mlp.fit(train_samples_shuffled, cluster_gen_labels_shuffled)
	debug.write("(Final iteration - number: " + str(mlp.n_iter_) + ", loss: " + str(mlp.loss_) + ")\n")
	pred_nn = mlp.predict(test_samples)
	debug.write("Tried Neural Network. ")
	
	# Try SVM Classifier
	svm_model = svm.SVC(probability=True, C=1000,  kernel='linear')
	svm_model.fit(training_samples, cluster_gen_labels)
	pred_svm = svm_model.predict(test_samples)
	debug.write("Tried SVM. ")

	# Try Decision Tree
	dt = RandomForestClassifier().fit(training_samples, cluster_gen_labels)
	pred_dt = dt.predict(test_samples)
	debug.write("Tried Decision Tree. ")

	# Try Naive Bayes
	nb = GaussianNB().fit(training_samples, cluster_gen_labels)
	pred_nb = nb.predict(test_samples)
	debug.write("Tried Naive Bayes.\n\n\n")

	debug.write("Exp: " + str(test_labels) + "\n")
	debug.write("--------------------------------------------------------------------------------------------------------------------------\n")
	debug.write("NeN: " + str(pred_nn) + "\n")
	debug.write("SVM: " + str(pred_svm) + "\n")
	debug.write("DeT: " + str(pred_dt) + "\n")
	debug.write("NaB: " + str(pred_nb) + "\n\n\n")

	return [pred_nn, pred_svm, pred_dt, pred_nb]

