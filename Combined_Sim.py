import Properties as pr
import Utils as u
import SimulationUtils as simu
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

debug = open('output/debug/combined_simulation.txt', 'w')
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=200)

def run_combined_sim():
	print_simulation_properties()
	
	overall_results = np.zeros((len(pr.classifiers), len(pr.comb_values_to_test)))

	for i in range(0, len(pr.comb_values_to_test)):
		debug.write("***********************\n")
		debug.write("Combined results with accuracy: " + str(pr.comb_values_to_test[i]) + ".\n\n\n")
		debug.write("---------------------------------------------------\n\n")
		print("\n==============================================\n")
		print("Starting simulation with accuracy: " + str(pr.comb_values_to_test[i]) + ".\n")

		# CHANGE ME - Whichever metric is being tested
		############################################################################
		pr.malicious_mu = pr.comb_values_to_test[i]
		############################################################################
		
		simu.update_properties()
		
		value_counts = np.zeros(len(pr.classifiers))
		conf_matrix_counts = np.zeros((len(pr.classifiers), 4))
		
		start = time.time()
		for j in range(0, pr.number_of_runs):
			print ("Run " + str(j+1) + ".")

			# CHANGE ME - Whichever metrics need to be calculated as per 'values_to_plot'
			############################################################################
			# Retry until valid cluster is achieved - unless 'allow invalid clusters' is true - usually succeeds first try
			valid_cluster = False
			while not (valid_cluster):
				gen_data = simu.generate_data('Training')	
				gen_samples = gen_data[0]
				gen_labels = gen_data[1]

				dbscan_labels = simu.cluster_data(gen_samples)
				
				if len(set(dbscan_labels)) >= 2 or pr.allow_invalid_clusters:
					valid_cluster = True
				else:
					print ("Retrying Clustering..")
					debug.write("\nClustering Invalid. Retrying.")

			gen_test_data = simu.generate_data('Testing')	
			gen_test_samples = gen_test_data[0]
			gen_test_labels = gen_test_data[1]

			debug.write(str(gen_labels) + "\n")			
			debug.write(str(dbscan_labels) + "\n\n\n")

			predictions = simu.classify(gen_samples, gen_test_samples, gen_test_labels, dbscan_labels, debug)
			
			# For each classifier prediction; NN, SVM, DT, NB. Calculate accuracy of classification.
			table = PrettyTable(['Classifier', 'Accuracy'])
			for k in range(0, len(predictions)):
				class_acc, clust_recall, clust_prec = simu.calculate_metrics(gen_test_labels, predictions[k])
				value_counts[k] += class_acc
				if pr.comb_output == "conf_matrix":
					print(confusion_matrix(gen_test_labels, predictions[k]))
					tn, fp, fn, tp = confusion_matrix(gen_test_labels, predictions[k]).ravel()
					conf_matrix_counts[k, 0] += tn
					conf_matrix_counts[k, 1] += fp
					conf_matrix_counts[k, 2] += fn
					conf_matrix_counts[k, 3] += tp
				table.add_row([pr.classifiers[k], str(class_acc)])
			############################################################################

			debug.write(str(table) + "\n\n")
			debug.write("==============================================\n")

		end = time.time()
		debug.write("Simulation for accuracy " + str(pr.comb_values_to_test[i]) + " finished. Run time: " + str(round(end-start, 2)) + "s, Total " + str(pr.number_of_runs) + " runs.\n")
		debug.write("***********************\n\n\n")

		# Get averages of all values
		for k in range(0, len(pr.classifiers)):
			if pr.comb_output == "graph":
				overall_results[k, i] = round(value_counts[k]/pr.number_of_runs, 4)
			elif pr.comb_output == "conf_matrix":
				for l in range(0, 4):
					conf_matrix_counts[k, l] = round(conf_matrix_counts[k, l] / pr.number_of_runs, 0)
	
	if pr.comb_output == "graph":
		graph_results(overall_results)
	elif pr.comb_output == "conf_matrix":
		confusion_matrix_results(conf_matrix_counts)
		
	print ("Done.")

# Method which calls which graphs to graph. Comment out ones to disable.
def graph_results(overall_results):
	# Pickle Dump Data to be used again if needed
	pickle.dump([overall_results, pr.classifiers, pr.comb_values_to_test, pr.class_test_values_label, 'Classifier'], \
		open('output/saved-results/clustering_accuracy_results.pkl', 'wb'))
	# Plot bar graph
	title = 'Performance of Classification for Clusters using Different ' + pr.clust_test_values_label
	subtitle = 'Samples: ' + str(pr.number_of_users) + ', Normal (mu, sigma): (' + str(pr.normal_mu) + ', ' \
		+ str(pr.normal_sigma) + '), Malicious Sigma: ' + str(pr.malicious_sigma) + ', Runs: ' + str(pr.number_of_runs) + ', Generated Data' 
	shplot.plot_multi_bar(overall_results, pr.classifiers, pr.comb_values_to_test, pr.comb_test_values_label, 'Malicious Mean', title, subtitle, \
		'Classifier', 'combined_accuracy_bar')

def confusion_matrix_results(conf_matrix_results):
	for i in range(0, len(conf_matrix_results)):
		debug.write(str([[conf_matrix_results[i][0], conf_matrix_results[i][1]], [conf_matrix_results[i][2], conf_matrix_results[i][3]]]) + "\n\n")
		shplot.plot_confusion_matrix([[conf_matrix_results[i][0], conf_matrix_results[i][1]], [conf_matrix_results[i][2], conf_matrix_results[i][3]]], pr.classifiers[i])


def generate_ideal_cluster_data():
	max_gen_data = 0
	max_acc = 0
	for i in range(0, pr.cluster_best_out_of):
		# Generate Data
		gen_data = simu.generate_data('Training')	
		gen_samples = gen_data[0]
		gen_true_labels = gen_data[1]
		# Cluster Data
		dbscan_labels = cluster_data(gen_samples)
		# Get Clustering Accuracy
		clust_acc, clust_recall, clust_prec = simu.calculate_metrics(gen_true_labels, dbscan_labels)

		if clust_acc > max_acc:
			max_acc = clust_acc
			max_gen_data = gen_data
	
	return max_gen_data, dbscan_labels
			

def print_simulation_properties():
	debug.write("=== Original Conditions: ===" + "\n")
	debug.write("Normal mean: " + str(pr.normal_mu) + "\n")
	debug.write("Normal sigma: " + str(pr.normal_sigma) + "\n")
	debug.write("Malicious mean: " + str(pr.malicious_mu) + "\n")
	debug.write("Malicious sigma: " + str(pr.malicious_sigma) + "\n")
	debug.write("Adversary: " + pr.adversary + "\n")
	debug.write("Fraction of deviators: " + str(pr.fraction_of_deviators) + "\n")
	debug.write("Sensor index: " + str(pr.sensor_index) + "\n")
	debug.write("Number of runs: " + str(pr.number_of_runs) + "\n\n")
	debug.write("Data type: " + ("Real" if pr.real_data else "Generated") + "\n")
	debug.write("Number of users: " + str(pr.number_of_users) + "\n")
	debug.write("=> Normal: " + str(pr.number_of_trustworthy_users) + ", Deviating: " + str(pr.number_of_deviating_users) + "\n")
	debug.write("Number of test users: " + str(pr.number_of_test_users) + "\n")
	debug.write("=> Normal: " + str(pr.number_of_trustworthy_test_users) + ", Deviating: " + str(pr.number_of_deviating_test_users) + "\n")
	debug.write("===================\n\n")
	debug.write("=== Changing Conditions: ===" + "\n")
	debug.write("Test Accuracies: " + str(pr.class_values_to_test) + "\n")
	debug.write("Using " + str(pr.class_gen_values_label) + " values " + str(pr.class_gen_values) + " to obtain accuracies.\n")
	debug.write("Classifiers Used: " + str(pr.classifiers) + "\n")
	debug.write("===================\n\n\n")
	


if __name__ == "__main__":
	run_combined_sim()

