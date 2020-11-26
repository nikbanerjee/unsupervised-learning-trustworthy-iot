__author__ = 'nikbanerjee'
from matplotlib import pyplot as plt
import numpy as np
import Properties as pr


# Plot Line Graph with single line
def plot_line(saveas, results, test_means, measurement_metric):
	plt.rc('font', family='Georgia')
	fig, ax = plt.subplots()
	
	plt.plot(results, color='r', label=measurement_metric, linewidth=2)
	plt.xticks(np.arange(len(test_means)), test_means)

	plt.xlabel('Malicious Mean')
	plt.ylim(0.0, 1.0)
	plt.ylabel('Clustering ' + measurement_metric)
	suptitle = plt.suptitle(str(measurement_metric) + ' of Clustering for Different Malicious Normal Distribution Means', fontsize=12)
	title = 'Normal Mean, Sigma: (' + str(pr.normal_mu) + ', ' + str(pr.normal_sigma) + '), Malicious Sigma: ' + str(pr.malicious_sigma) + ', Samples: ' + \
		str(pr.number_of_users) + ', Deviators: ' + str(round(pr.fraction_of_deviators, 4)*100) + '%, Runs: ' + str(pr.number_of_runs)
	if pr.real_data:
		title = title + ', Real Data'
	else:
		title = title + ', Generated Data'
	plt.title(title, fontsize=8)
	lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig("output/graphs/" + str(saveas), bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')



# Plot Line Graph with multiple lines
def plot_multi_line(saveas, all_results, result_labels, x_labels, measurement_metric, linetype='Rough'):
	plt.rc('font', family='Georgia')
	fig, ax = plt.subplots()

	if linetype == 'Smooth':
		# For smooth line graph
		for i in range(0, len(all_results)):
			x_values = np.array(x_labels)
			y_values = np.array(all_results[i])
			x_smooth = np.linspace(x_values.min(), x_values.max(), 100)
			y_smooth = spline(x_values, y_values, x_smooth)
			plt.plot(x_smooth, y_smooth, color=pr.color_scheme[i], label=result_labels[i], linewidth=1.5)
			plt.xticks(np.arange(x_values.min(), x_values.max()+0.1, 1.0))
	else:
		# For rough line graph
		for i in range(0, len(all_results)):
			plt.plot(all_results[i], color=pr.color_scheme[i], label=result_labels[i], linewidth=1.5)
			plt.xticks(np.arange(len(x_labels)), x_labels)

	plt.xlabel('Clustering Accuracy')
	plt.ylim(0.0, 1.0)
	plt.ylabel('Classification ' + measurement_metric)
	suptitle = plt.suptitle(str(measurement_metric) + ' of Classification Based on Clustering Accuracy', fontsize=12)
	title = 'Normal Mean, Sigma: (' + str(pr.normal_mu) + ', ' + str(pr.normal_sigma) + '), Malicious Sigma: ' + str(pr.malicious_sigma) + ', Samples: ' + \
		str(pr.number_of_users) + ', Deviators: ' + str(round(prpr.files.fraction_of_deviators, 4)*100) + '%, Runs: ' + str(pr.number_of_runs)
	if pr.real_data:
		title = title + ', Real Data'
	else:
		title = title + ', Generated Data'
	plt.title(title, fontsize=8)
	lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.savefig("output/graphs/" + str(saveas), bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')


def plot_multi_bar(results, result_labels, x_labels, x_labelname, y_labelname, title, subtitle, legend_title, saveas):
	plt.rc('font', family='Georgia')
	index = np.arange(len(x_labels))
	fig, ax = plt.subplots()
	number_of_bars = len(results)
	bar_width = pr.bar_width_options[number_of_bars-1]

	start_offset = -1 * round((((number_of_bars * (bar_width + 0.05)) - 0.05) / 2), 3)

	for i in range(0, number_of_bars):
		offset = round(start_offset + (i * (bar_width + 0.05)), 3)
		plt.bar(index+offset, results[i], bar_width, label=result_labels[i], color=pr.color_scheme[i], alpha=0.7)
	
	
	plt.xlabel(x_labelname, fontsize=20)
	plt.xticks(np.arange(len(x_labels)), x_labels)
	plt.ylim(0.0, 1.0)
	plt.ylabel(y_labelname, fontsize=20)
	suptitle = plt.suptitle(title, fontsize=12)
	lgd = plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1.0), loc=2.0, borderaxespad=0.)
	
	# Save official EPS version for paper - without title
	plt.savefig("output/graphs/" + str(saveas), bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')
	# Save official PDF version for paper - without title
	plt.savefig("output/graphs/" + str(saveas) + ".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.title(subtitle, fontsize=10)
	# Save official EPS version for paper - with title
	plt.savefig("output/graphs/" + str(saveas) + "_title", bbox_extra_artists=(lgd,), bbox_inches='tight', format='eps')
	# Save annotated version for professors
	plt.savefig("output/graphs/" + str(saveas) + "_prof", bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight')



def plot_confusion_matrix(conf_matrix, classifier):
	plt.rc('font', family='Georgia')
	fig, ax = plt.subplots()
	plt.matshow(conf_matrix)
	plt.colorbar()
	plt.title(classifier + " Confusion Matrix", fontsize=16)
	plt.ylabel('Actual', fontsize=20)
	plt.xlabel('Predicted', fontsize=20)
	plt.savefig('output/graphs/confusion_matrix_' + classifier, format='eps')
	plt.savefig('output/graphs/confusion_matrix_' + classifier, format='pdf')

