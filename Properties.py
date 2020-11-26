import math

# Clustering Simulation Only
#######################################################################################
# Values to test - x axis for graph. Note values will replace their values below.
clust_values_to_test = [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
clust_test_values_label = 'Malicious Sigma'

# Values to plot - Each bar or line on the graph. 
clust_values_to_plot = [16.5, 17, 17.5]
clust_plot_values_label = 'Malicious Mean'
clust_values_require_unique_runs = True

# Discard clustering results with no outliers found
allow_invalid_clusters = False
#######################################################################################

# Classifying Simulation Only
#######################################################################################
# Values to test - x axis for graph. Note values will replace their values below.
class_values_to_test = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
class_test_values_label = 'Clustering Accuracy'

# Metric and values used to generate data for the above values to test (ie. malicious mu for test accuracy).
class_gen_values = [16.4, 16.6, 16.8, 17, 17.5, 18, 18.5]
class_gen_values_label = 'Malicious Mean'

# Values to plot - Each bar or line on the graph. 
classifiers = ['Neural Net', 'SVM', 'Random Forest', 'Naive Bayes']
#######################################################################################

# Combined Simulation Only
#######################################################################################
# Values to test - x axis for graph. Note values will replace their values below.
#comb_values_to_test = [16.4, 16.6, 16.8, 17, 17.5, 18, 19]	# For graph
comb_values_to_test = [21]					# For conf_matrix (change sigma to 3 and data to REAL)
comb_test_values_label = 'Malicious Mean'

# Values to plot - Each bar or line on the graph. 
classifiers = ['Neural Net', 'SVM', 'Random Forest', 'Naive Bayes']

# Output - 'graph' or 'conf_matrix'
comb_output = "conf_matrix"
#######################################################################################

# Normal distribution stats - variables in the simulation.
normal_mu, normal_sigma = 16, 2
malicious_mu, malicious_sigma = 21, 3

# Percentage of malicious users in dataset
fraction_of_deviators = 0.40

# DBSCAN Epsilon
epsilon =  30

# Number of times to repeat experiment to obtain average - Increases run time
number_of_runs = 25

# Use real or generated data
real_data = False

if real_data:
	real_dataset_size = 40
	real_data_training_size = 20
	real_data_testing_size = real_dataset_size - real_data_training_size

	# Calculate number of users that will make the number of trustworthy users equal to the real data set
	number_of_users = math.ceil(real_data_training_size / (1-fraction_of_deviators))
	number_of_test_users = math.ceil(real_data_testing_size / (1-fraction_of_deviators))

else:
	# Generated dataset (train & test) size
	number_of_users = 100
	number_of_test_users = 50


# Using fraction of deviators to find number of malicious and trustworthy users
number_of_trustworthy_users = int((1-fraction_of_deviators)*number_of_users)
number_of_deviating_users = number_of_users - number_of_trustworthy_users

number_of_trustworthy_test_users = int((1-fraction_of_deviators)*number_of_test_users)
number_of_deviating_test_users = number_of_test_users - number_of_trustworthy_test_users


# Graphing
bar_width_options = [0.3, 0.25, 0.2, 0.1, 0.05]
color_scheme = ['r', 'b', 'orange', 'g']


# Other
number_of_measurements = 1000
random = False
adversary = 'normal'
sensor_index = 5


# TODO - SORT
largest_cluster = -1
plot = False
performance_test = True
regions = 20
theta = 0
std = 0
bins = []
hist = []
max_distance = float('-inf')
max_bin = -1
budget = 100
