import Properties as pr
import Utils as u
import SimulationUtils as simu
import sys
import numpy as np
import scipy as sp
from scipy.interpolate import spline
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

debug = open('output/debug/classifying_simulation.txt', 'w')
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=200)

def run_classifying_sim():
	print_simulation_properties()
	
	overall_results = np.zeros((len(pr.classifiers), len(pr.class_values_to_test)))

	for i in range(0, len(pr.class_values_to_test)):
		debug.write("***********************\n")
		debug.write("Classifying with accuracy: " + str(pr.class_values_to_test[i]) + ", malicious mean: " + str(pr.class_gen_values[i]) + ".\n\n\n")
		debug.write("---------------------------------------------------\n")
		print("\n==============================================\n")
		print("Starting simulation with accuracy: " + str(pr.class_values_to_test[i]) + ", malicious mean: " + str(pr.class_gen_values[i]) + ".\n")
		
		value_counts = np.zeros(len(pr.classifiers))
		
		start = time.time()
		for j in range(0, pr.number_of_runs):
			print ("Run " + str(j+1) + ".")

			# CHANGE ME - Whichever metric is being tested
			############################################################################
			pr.malicious_mu = pr.class_gen_values[i]
			############################################################################
			
			simu.update_properties()

			gen_data = simu.generate_data('Training')
			gen_test_data = simu.generate_data('Testing')
			training_samples = gen_data[0]
			test_samples = gen_test_data[0]
			test_labels = gen_test_data[1]

			# Gen cluster labels
			cluster_gen_labels = simu.generate_cluster_labels(pr.number_of_users, pr.class_values_to_test[i])
			
			# CHANGE ME - Whichever metrics need to be calculated as per 'values_to_plot'
			############################################################################

			# Classify with different classifiers
			predictions = simu.classify(training_samples, test_samples, test_labels, cluster_gen_labels, debug)
			
			# For each classifier prediction; NN, SVM, DT, NB. Calculate accuracy of classification.
			table = PrettyTable(['Classifier', 'Accuracy'])
			for k in range(0, len(predictions)):
				class_acc, clust_recall, clust_prec = simu.calculate_metrics(test_labels, predictions[k])
				value_counts[k] += class_acc
				table.add_row([pr.classifiers[k], str(class_acc)])
			############################################################################

			debug.write(str(table) + "\n\n")
			debug.write("==============================================\n")

		end = time.time()
		debug.write("Simulation for accuracy " + str(pr.class_values_to_test[i]) + " finished. Run time: " + str(round(end-start, 2)) + "s, Total " + str(pr.number_of_runs) + " runs.\n")
		debug.write("***********************\n\n\n")

		# Get averages of all values
		for k in range(0, len(pr.classifiers)):
			overall_results[k, i] = round(value_counts[k]/pr.number_of_runs, 4)
		
	graph_results(overall_results)
	print ("Done.")


def graph_results(overall_results):
	# Pickle Dump Data to be used again if needed
	pickle.dump([overall_results, pr.classifiers, pr.class_values_to_test, pr.class_test_values_label, 'Classifier'], \
		open('output/saved-results/classification_accuracy_results.pkl', 'wb'))
	# Plot bar graph
	title = 'Accuracy of Different Classifiers based on Clustering Accuracy'
	subtitle = 'Samples: ' + str(pr.number_of_users) + ', Normal (mu, sigma): (' + str(pr.normal_mu) + ', ' \
		+ str(pr.normal_sigma) + '), Malicious Sigma: ' + str(pr.malicious_sigma) + ', Deviators: ' + str(pr.fraction_of_deviators) + ', Runs: ' + str(pr.number_of_runs) + ', Generated Data' 
	shplot.plot_multi_bar(overall_results, pr.classifiers, pr.class_values_to_test, pr.class_test_values_label, 'Classification Accuracy', title, subtitle, \
		'Classifier', 'classification_accuracy_bar')


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
	run_classifying_sim()


