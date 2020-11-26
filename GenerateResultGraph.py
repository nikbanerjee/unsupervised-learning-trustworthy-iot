import Properties as pr
import sys
import numpy as np
import pickle
import CustomPlot as cplot


def generate_graph_from_pickled_results():

	### CHANGE ME ###
	pickle_file = "output/saved-results/combined_accuracy_results.pkl"
	title = "GRAPH TITLE"
	subtitle = "GRAPH SUBTITLE"
	y_label = "Classification Accuracy"
	save_as = "combined_accuracy"
	graph_type = "bar"	# 'bar' or 'conf_matrix'
	#################
	
	saved_data = pickle.load(open(pickle_file, 'rb'))
	if graph_type == "bar":
		cplot.plot_multi_bar(saved_data[0], saved_data[1], saved_data[2], saved_data[3], y_label, title, subtitle, saved_data[4], save_as + "_restored")
	elif graph_type == "conf_matrix":
		for i in range(0, len(conf_matrix_results)):
			cplot.plot_confusion_matrix([[saved_data[0][i][0], saved_data[0][i][1]], [saved_data[0][i][2], saved_data[0][i][3]]], saved_data[1][i])


def main():
	generate_graph_from_pickled_results()


if __name__ == "__main__":
	main()
