import Properties as pr
import Utils as u
import sys
import numpy as np
import pickle

np.set_printoptions(threshold=sys.maxsize)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Returns array containing [0] Samples stacked, [1] True Labels Stacked.
def generate_samples(stage='Train'):
	if stage != 'Train' and stage != 'Test':
		print ("\nStage must be 'Train' or 'Test', was '" + stage + "'.\n")
		sys.exit()
	total_sample_count = (pr.number_of_test_users if stage == 'Test' else pr.number_of_users)
	normal_sample_count = int((1-pr.fraction_of_deviators)*int(total_sample_count))
	deviating_sample_count = int(pr.fraction_of_deviators*int(total_sample_count))
	outfile = 'testSamples' if stage == 'Test' else 'trainSamples'
	debug = open('debug/' + outfile + '.txt', 'w')

	print ("\nGenerating " + stage + "ing data..\n")

	# Generate normal samples from distribution
	if pr.real_data:
		u.load_files()
		normal_samples = u.read_samples('sensors', pr.sensor_index, 0, pr.number_of_users)
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
		deviating_samples = u.generate_samples_random(pr.number_of_measurements, pr.test_data_deviating)

	print ("Stacking data..\n")
	samples = np.vstack((normal_samples, deviating_samples))
	labels = np.hstack((np.zeros(len(normal_samples), dtype=np.int), np.ones(len(deviating_samples), dtype=np.int)))

	debug.write("------------SAMPLES---------------" + "\n")
	debug.write(str(samples) + "\n")
	debug.write("LENGTH: " + str(len(samples)) + "x" + str(len(samples[0])) + "\n\n")
	debug.write("------------TRUE LABELS---------------" + "\n")
	debug.write(str(labels) + "\n")
	debug.write("LENGTH: " + str(len(labels)) + "\n\n")

	print ("Pickling data to file (" + outfile + ".pkl)..\n")
	pickle.dump([samples, labels], open(outfile + '.pkl', 'wb'))
	
	sample_size = (pr.number_of_test_users if stage == 'Test' else pr.number_of_users)
	print ("Finished. Generated " + str(sample_size) + " samples.\n")


def main():
	if len(sys.argv) == 2:
		generate_samples(stage=sys.argv[1])
	else:
		generate_samples()


if __name__ == "__main__":
	main()
	
	
