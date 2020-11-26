from dependencies.pyds import MassFunction
import numpy as np
from os import listdir
from os.path import isfile, join
import csv
import scipy as sp
import Properties as pr
import string
from collections import Counter
from random import shuffle
import sys

def generate_mass(readings):
    """Computes a mass function that can be used by DS.
    :param readings: the sensory readings.
    :param values: the intervals
    """
    '''
    The idea here is that we have one symbol representing each interval. To generate masses easier
    we map each symbol to a number.
    '''
    char_values = list(string.printable)
    int_values  = [i for i in range(0,len(char_values))]
    mappings = dict(zip(int_values, char_values))
    measurements = []

    for i in mappings.keys():
        count = sum(1 for x in readings if x >= i and x < i+1)/len(readings)
        measurements.append(count)
    return measurements

def generate_samples(mu, sigma, samples, number_of_users):
    """Computes a number of sensory readings according to a normal distribution. Then it uses generate_mass to generate DS masses
    :param mu: the mean of the normal distribution.
    :param sigma: the std
    :param how meany readings per user
    :param number_of_users: the number of users
    """
    masses = []
    # Deviating Values
    for i in range(0, number_of_users):
        np.random.seed()
        s = np.random.normal(mu, sigma, samples)
        user_mass = generate_mass(s)
        if i == 0:
            masses = user_mass
        else:
            masses= np.vstack([masses, user_mass])
    return masses

def generate_samples_gamma(mu, skew, samples, number_of_users):
    """Computes a number of sensory readings according to a gamma distribution. Then it uses generate_mass to generate DS masses
    :param mu: the mean of the normal distribution.
    :param sigma: the std
    :param how meany readings per user
    :param number_of_users: the number of users
    """
    masses = []
    # Deviating Values
    for i in range(0, number_of_users):
        np.random.seed()
        s = np.random.gamma(mu, skew, samples)
        user_mass = generate_mass(s)
        if i == 0:
            masses = user_mass
        else:
            masses= np.vstack([masses, user_mass])
    return masses

def generate_samples_uniform(samples, number_of_users):
    """Computes a number of sensory readings according to a uniform distribution. Then it uses generate_mass to generate DS masses
    :param how meany readings per user
    :param number_of_users: the number of users
    """
    pr.random  =True
    masses = []
    # Deviating Values
    for i in range(0, number_of_users):
        np.random.seed()
        s = np.random.randint(0,100, samples)
        user_mass = generate_mass(s)
        if i == 0:
            masses = user_mass
        else:
            masses= np.vstack([masses, user_mass])
    return masses


def generate_samples_random(samples, number_of_users):
    """Computes a number of sensory readings according to a uniform distribution. Then it uses generate_mass to generate DS masses
    :param how meany readings per user
    :param number_of_users: the number of users
    """
    pr.random  =True
    masses = []
    # Deviating Values
    for i in range(0, number_of_users):
        np.random.seed()
        bins = np.random.randint(1,100)
        s=np.zeros(bins)
        for b in range(bins):
            np.random.seed()
            s[b] = np.random.randint(0,100)
        user_mass = generate_mass(s)
        if i == 0:
            masses = user_mass
        else:
            masses= np.vstack([masses, user_mass])
    return masses

def load_files():
    pr.files = [ f for f in listdir('sensors') if isfile(join('sensors',f)) ]
    shuffle(pr.files)


def read_samples(folder, index_of_desired_measurement, file_start, file_end, should_print=False):
    """Reads the samples from the files in the specified folder. Each file should represent one sensor.
    :param folder: the folder containing the files
    :param index_of_desired_measurement: the index in the csv of the desired measurement
    :param values: the intervals
    """
    # since we get a folder with different measurements we list all files
    masses = []
    #readingsfile = open('readings.txt', 'w')
    #massesfile = open('massesfile.txt', 'w')
    files = pr.files[file_start:file_end]
    i = 0
    #readingsfile.write("************* READINGS **************\n")
    #massesfile.write("************* MASSES **************\n")
    #np.set_printoptions(threshold=sys.maxsize)

    for f in files:
        #print("READING FILE: " + f)
        #readingsfile.write("==== " + f + " ====\n")
        #massesfile.write("==== " + f + " ====\n")
        path = folder+"/"+f
        ifile  = open(path, "rt")
        reader = csv.reader(ifile, delimiter=',')
        readings = []
        k = 0
        for row in reader:
            if k == 0:
                k = 1
                continue
            value = row[index_of_desired_measurement]
            if should_print:
                print(value)
            if value != '':
                readings.append(float(value))
        if len(readings)==0:
            continue

        #readingsfile.write(str(readings) + "\n")
        m = generate_mass(readings)
        #massesfile.write(str(m) + "\n")
        if i == 0:
            masses = m
            i = 1
        else:
            masses= np.vstack([masses, m])

    return masses

def combine_masses(samples):
    """Uses Schafers combination rule to combine the masses
    :param masses: the probability masses
    :param values: the intervals
    """
    combined_mass = None
    for i in range(len(samples)-1):

        dict_x = dict(zip( list(string.printable) , samples[i][:]))
        dict_y = dict(zip( list(string.printable) , samples[i+1][:]))

        dict_x = {key: value for key, value in dict_x.items() if value != 0.0}
        dict_y = {key: value for key, value in dict_y.items() if value != 0.0}

        mx = MassFunction(dict_x)
        my = MassFunction(dict_y)

        # here we use Dempsters combination rule
        combined_mass = mx&my
    return combined_mass

def print_clusters(labels, X):
    """Prints information regarding the clusters generated by DBSCAN
    :param DBSCAN: the DBSCAN classifier
    :param X: the probability masses
    """
    for k in np.unique(labels):
        members = np.where(labels == k)[0]
        if k == -1:
            print("outliers:",members,len(X[members]))
            #print(X[members])
            print()
        else:
            print("cluster %d:" % k,members,len(X[members]))
            #print(X[members])
            print()

def largest_cluster(labels):
    # first find the cluster with the fewest members
    c = Counter(labels)
    max = 0
    k = -1

    for key in c.keys():
        if max < c.get(key) and key !=-1:
            max = c.get(key)
            k = key
    pr.largest_cluster = k

def remove_outliers(labels, X):
    """Removes the outliers from the samples and returns the filtered probability masses
    :param labels: the classification of the  classifier
    :param X: the probability masses
    """
    #print("Before: ", X)
    filtered = []
    first = True
    for k in range(0,len(labels)):
        if labels[k] != pr.largest_cluster:
            continue
        else:
            if first:
                filtered =  X[k][:]
                first = False
            else:
                filtered = np.vstack([filtered, X[k][:]])
    #print("After: ",filtered)
    return filtered


def distance_ds_conflict(X,Y):
    """Computes the distances of 2 samples (masses) as the DS weight of conflict
    :param X, Y: the 2 probability masses
    """
    # make mass functions
    dict_x = dict(zip( list(string.printable) , X))
    dict_y = dict(zip( list(string.printable) , Y))

    mx = MassFunction(dict_x)
    my = MassFunction(dict_y)

    mx_features =  [get_belief_plausibility(mx)]
    my_features =  [get_belief_plausibility(my)]
    mx_features = np.vstack([mx_features, my_features])
    distance = sp.spatial.distance.pdist(mx_features,'canberra')
    return distance

def get_map(value):
    """Returns the symbol representing the defined numerical value
    :param value: numerical value
    """
    char_values = list(string.printable)
    int_values  = [i for i in range(0,len(char_values))]
    mappings = dict(zip(int_values, char_values))
    return mappings[value]

def get_key(key):
    """Returns the symbol representing the defined numerical value
    :param value: numerical value
    """
    char_values = list(string.printable)
    int_values  = [i for i in range(0,len(char_values))]
    mappings = dict(zip(char_values, int_values))
    return mappings[key]

def get_belief_plausibility(mass):
    """Returns belief and the plausibility for the given value
    :param value: numerical value
    :param mass: the mass
    """

    max_bel = 0
    max_index = 0
    max_pl = 0
    for i in list(string.printable):
        if mass.bel(i) > max_bel:
            max_bel = mass.bel(i)
            max_pl = mass.pl(i)
            max_index = get_key(i)
    return max_bel, max_index, mass.local_conflict()



def plot_clusters(labels, X, core_samples):
    import pylab as pl
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    extremes = []

    fig = pl.figure()
    ax = Axes3D(fig)
    ax.set_xlabel(r'$H_{max}$', fontsize=18)
    ax.set_ylabel(r'$Bel(H_{max})$', fontsize=18)
    ax.set_zlabel(r'$LCon$', fontsize=18, rotation=90)

    marker_styles = ['o','*','^','d']
    points = []
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
            mark = 'v'
            markersize = 70
        else:
            mark = marker_styles[k]
        class_members = [index[0] for index in np.argwhere(labels == k)]
        cluster_core_samples = [index for index in core_samples if labels[index] == k]
        first = True
        for index in class_members:
            x = X[index]
            if index in core_samples and k != -1:
                markersize = 400
            else:
                markersize = 80
            dict_x = dict(zip( list(string.printable) , x))
            mx = MassFunction(dict_x)
            m, index, local  = get_belief_plausibility(mx)
            ax.scatter(index, m, local, marker=mark, s=markersize, c=col)
            if first:
                points.append([m,index,local])
        ax.zaxis.set_rotate_label(False)
        pl.savefig("cluster.pdf", format="pdf", ext="pdf", close=False, verbose=True)


