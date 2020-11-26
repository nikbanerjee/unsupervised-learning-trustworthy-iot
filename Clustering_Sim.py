import Properties as pr
import Utils as u
import SimulationUtils as simu
import sys
import math
import pickle
import numpy as np
import scipy as sp
import time
import CustomPlot as shplot
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix

debug = open('output/debug/clustering_simulation.txt', 'w')
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=200)

def run_clustering_sim():	
	print_simulation_properties()
	print("")

	overall_results = np.zeros((len(pr.clust_values_to_plot), len(pr.clust_values_to_test)))
	
	for i in range(0, len(pr.clust_values_to_test)):
		debug.write("***********************\n")
		debug.write("Clustering with " + pr.clust_test_values_label + ": " + str(pr.clust_values_to_test[i]) + ".\n\n")
		print("=========================================")
		print("Starting simulation with " + pr.clust_test_values_label + ": " + str(pr.clust_values_to_test[i]) + ".\n")
		
		start = time.time()
		number_of_plot_runs = (len(pr.clust_values_to_plot) if pr.clust_values_require_unique_runs else 1)
		value_counts = np.zeros(len(pr.clust_values_to_plot))
		
		for j in range(0, number_of_plot_runs):
			for k in range(0, pr.number_of_runs):
				iter_print = "Run " + str(k+1)
				if pr.clust_values_require_unique_runs: iter_print += ", Plot " + str(j+1) + " (" + str(pr.clust_plot_values_label) + ": " + str(pr.clust_values_to_plot[j]) + ")."
				print (iter_print)

				# CHANGE ME - Whichever metric is being tested
				############################################################################
				pr.malicious_sigma = pr.clust_values_to_test[i]
				pr.malicious_mu = pr.clust_values_to_plot[j]
				############################################################################
			
				simu.update_properties()
	
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
					

				# CHANGE ME - Whichever metrics need to be calculated as per 'values_to_plot'
				############################################################################
				clust_acc, clust_recall, clust_precision = simu.calculate_metrics(gen_labels, dbscan_labels)
				
				if pr.clust_values_require_unique_runs:
					value_counts[j] += clust_acc
					debug_message = pr.clust_test_values_label + ": " + str(pr.clust_values_to_test[i]) + ", Cluster accuracy: " + str(round(clust_acc, 4)*100) + "%."
				else:
					value_counts[0] += clust_acc
					value_counts[1] += clust_recall
					value_counts[2] += clust_precision
					debug_message = pr.clust_test_values_label + ": " + str(pr.clust_values_to_test[i]) + ", Cluster accuracy: " + str(round(clust_acc, 4)*100) \
						+ "%, Percent of outliers found: " + str(round(clust_recall, 4)*100) + "%, Precision: " + str(round(clust_recall, 4)*100) + "%."
				############################################################################


				debug.write("\n" + str(debug_message) + "\n\n")
				debug.write(str(gen_labels) + "\n")
				debug.write(str(dbscan_labels) + "\n\n")
				print (debug_message)
				print ("***********************\n")


		# Average things
		for j in range(0, len(pr.clust_values_to_plot)):
			overall_results[j, i] = round(value_counts[j]/pr.number_of_runs, 4)
		
		end = time.time()
		debug.write("Simulation finished. Run time: " + str(round(end-start, 2)) + "s, Total " + str(pr.number_of_runs) + " runs, " + str(number_of_plot_runs) + " plots.\n")
		debug.write("***********************\n\n\n")
	
	graph_results(overall_results)
	print ("Done.")



# Method which calls which graphs to graph. Comment out ones to disable.
def graph_results(overall_results):
	# Pickle Dump Data to be used again if needed
	pickle.dump([overall_results, pr.clust_values_to_plot, pr.clust_values_to_test, pr.clust_test_values_label, pr.clust_plot_values_label], \
		open('output/saved-results/clustering_accuracy_results.pkl', 'wb'))
	# Plot bar graph
	title = 'Performance of Clustering for Different ' + pr.clust_test_values_label + ' and ' + pr.clust_plot_values_label
	subtitle = 'Samples: ' + str(pr.number_of_users) + ', Normal (mu, sigma): (' + str(pr.normal_mu) + ', ' \
		+ str(pr.normal_sigma) + '), Malicious Sigma: ' + str(pr.malicious_sigma) + ', Runs: ' + str(pr.number_of_runs) + ', Epsilon: ' + str(pr.epsilon) + ', Generated Data' 
	shplot.plot_multi_bar(overall_results, pr.clust_values_to_plot, pr.clust_values_to_test, pr.clust_test_values_label, 'Clustering Accuracy', title, subtitle, \
		pr.clust_plot_values_label, 'clustering_accuracy')



def print_simulation_properties():
	debug.write("=== Original Conditions: ===" + "\n")
	debug.write("Normal mean: " + str(pr.normal_mu) + "\n")
	debug.write("Normal sigma: " + str(pr.normal_sigma) + "\n")
	debug.write("Malicious mean: " + str(pr.malicious_mu) + "\n")
	debug.write("Malicious sigma: " + str(pr.malicious_sigma) + "\n")
	debug.write("Adversary: " + pr.adversary + "\n")
	debug.write("Fraction of deviators: " + str(pr.fraction_of_deviators) + "\n")
	debug.write("Sensor index: " + str(pr.sensor_index) + "\n")
	debug.write("Number of runs: " + str(pr.number_of_runs) + "\n")
	debug.write("DBSCAN Epsilon: " + str(pr.epsilon) + "\n\n")
	debug.write("Data type: " + ("Real" if pr.real_data else "Generated") + "\n")
	debug.write("Number of users: " + str(pr.number_of_users) + "\n")
	debug.write("=> Normal: " + str(pr.number_of_trustworthy_users) + ", Deviating: " + str(pr.number_of_deviating_users) + "\n")
	debug.write("Number of test users: " + str(pr.number_of_test_users) + "\n")
	debug.write("=> Normal: " + str(pr.number_of_trustworthy_test_users) + ", Deviating: " + str(pr.number_of_deviating_test_users) + "\n")
	debug.write("===================\n\n")
	debug.write("=== Changing Conditions: ===" + "\n")
	debug.write("Metric to Test: " + str(pr.clust_test_values_label) + "\n")
	debug.write("Testing Values: " + str(pr.clust_values_to_test) + "\n")
	debug.write("Plot Labels (per test): " + str(pr.clust_values_to_plot) + "\n")
	debug.write("Unique Run per Plot Value: " + ("True" if pr.clust_values_require_unique_runs else "False") + "\n")
	debug.write("===================\n\n\n")


if __name__ == "__main__":
	run_clustering_sim()


