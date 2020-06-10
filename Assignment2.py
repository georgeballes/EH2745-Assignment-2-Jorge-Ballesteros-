# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:55:54 2020

@author: georg
"""
import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt

import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from collections import Counter


def simple_test_net():

    net = pp.create_empty_network()
    
    # I  am going to define here the buses which form the system. As there are no transformers,
    # the voltage is the same in every busbar
    Bus_Voltage = 110 # in kv
    
    Bus_1 = pp.create_bus(net, Bus_Voltage, name='Bus 1-CLARK')
    Bus_2 = pp.create_bus(net, Bus_Voltage, name='Bus 2-AMHERST')
    Bus_3 = pp.create_bus(net, Bus_Voltage, name='Bus 3-WINLOCK')
    Bus_4 = pp.create_bus(net, Bus_Voltage, name='Bus 4-BOWMAN')
    Bus_5 = pp.create_bus(net, Bus_Voltage, name='Bus 5-TROY')
    Bus_6 = pp.create_bus(net, Bus_Voltage, name='Bus 6-MAPLE')
    Bus_7 = pp.create_bus(net, Bus_Voltage, name='Bus 7-GRAND')
    Bus_8 = pp.create_bus(net, Bus_Voltage, name='Bus 8-WAUTAGA')
    Bus_9 = pp.create_bus(net, Bus_Voltage, name='Bus 9-CROSS')
    
    
    # Next step is create the generators of the system, we know from the assignment that
    # Bus 1 is the slack bus along with generation capable of supplying the system should
    # another generator fall out of operation.
    
    Generator1_Pg_MW = 0
    Generator2_Pg_MW = 163
    Generator3_Pg_MW = 85
    
    # Q generated is 0 for all generators
    
    Generator_1 = pp.create_gen(net, Bus_1, Generator1_Pg_MW, slack=True, name='Generator 1')
    Generator_2 = pp.create_sgen(net, Bus_2, Generator2_Pg_MW, q_mvar=0, name='Generator 2')
    Generator_3 = pp.create_sgen(net, Bus_3, Generator3_Pg_MW, q_mvar=0, name='Generator 3')


    # In order to create the loads we need to define the P and Q consumed by each load
    Load5_Pd_MW = 90
    Load5_Qd_MVAR = 30
    Load7_Pd_MW = 100
    Load7_Qd_MVAR = 35
    Load9_Pd_MW = 125
    Load9_Qd_MVAR = 50
    
    Load_5 = pp.create_load(net, Bus_5, Load5_Pd_MW , Load5_Qd_MVAR, name='Load 5')
    Load_7 = pp.create_load(net, Bus_7, Load7_Pd_MW , Load7_Qd_MVAR, name='Load 7')
    Load_9 = pp.create_load(net, Bus_9, Load9_Pd_MW , Load9_Qd_MVAR, name='Load 9')
    
    
    Line_length = 10 # in km
    Line_1_4 = pp.create_line(net, Bus_1, Bus_4, Line_length, '149-AL1/24-ST1A 110.0', name='Line 1 to 4')
    Line_2_8 = pp.create_line(net, Bus_2, Bus_8, Line_length, '149-AL1/24-ST1A 110.0', name='Line 2 to 8')
    Line_3_6 = pp.create_line(net, Bus_3, Bus_6, Line_length, '149-AL1/24-ST1A 110.0', name='Line 3 to 6')
    Line_4_5 = pp.create_line(net, Bus_4, Bus_5, Line_length, '149-AL1/24-ST1A 110.0', name='Line 4 to 5')
    Line_4_9 = pp.create_line(net, Bus_4, Bus_9, Line_length, '149-AL1/24-ST1A 110.0', name='Line 4 to 9')
    Line_5_6 = pp.create_line(net, Bus_5, Bus_6, Line_length, '149-AL1/24-ST1A 110.0', name='Line 5 to 6')
    Line_6_7 = pp.create_line(net, Bus_6, Bus_7, Line_length, '149-AL1/24-ST1A 110.0', name='Line 6 to 7')
    Line_7_8 = pp.create_line(net, Bus_7, Bus_8, Line_length, '149-AL1/24-ST1A 110.0', name='Line 7 to 8')
    Line_8_9 = pp.create_line(net, Bus_8, Bus_9, Line_length, '149-AL1/24-ST1A 110.0', name='Line 8 to 9')
    
    
    return net

def create_data_source(net, mode='', n_timesteps=30):
    profiles = pd.DataFrame()
    if mode == 'High Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 1.05 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 1.05 * net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) * net.load.q_mvar[i])
    elif mode == 'Low Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 0.90 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 0.90 *  net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) *  net.load.q_mvar[i])
    
    ds = DFData(profiles)

    return profiles, ds




# Aqui el ConstControl lo que va a hacer es relacionar nuestro sistema en pandapower
# y el datasource que acabamos de crear (funciona como un controller). element='load'
# significa que cogemos en pandapower el elemento load,la variable p o q y vemos segun
# su indice (tenemos 3 loads asi que de 0 a 2), despues cogem el datasource creado y 
# tienes que especificar su correspondiente profile_name del data source
def create_controllers(net, ds):
    for i in range(len(net.load)):
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=ds, profile_name=['load{}_P'.format(str(i))])
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=ds, profile_name=['load{}_Q'.format(str(i))])

    return net
    

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    return ow


net = simple_test_net()
n_time_steps = 60

def high_load(net, n_time_steps, output_dir):
    _net = net
    profiles, ds = create_data_source(_net, mode='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)


def low_load(net, n_time_steps, output_dir):
    _net = net
    profiles, ds = create_data_source(_net, mode='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)


def gen_discon_high(net, n_time_steps, output_dir):
    _net = net
    _net.sgen.in_service[1] = False
    profiles, ds = create_data_source(_net, mode='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[1] = True

    
def gen_discon_low(net, n_time_steps, output_dir):
    _net = net
    _net.sgen.in_service[1] = False
    profiles, ds = create_data_source(_net, mode='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[1] = True

def line_discon_high(net, n_time_steps, output_dir):
    _net = net
    index_line_5_6 = pp.get_element_index(_net, 'line', 'Line 5 to 6')
    _net.line.in_service[index_line_5_6] = False
    profiles, ds = create_data_source(_net,  mode='High Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line_5_6] = True

    
def line_discon_low(net, n_time_steps, output_dir):
    _net = net
    index_line_5_6 = pp.get_element_index(_net, 'line', 'Line 5 to 6')
    _net.line.in_service[index_line_5_6] = False
    profiles, ds = create_data_source(_net,  mode='Low Load', n_timesteps=n_time_steps)
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, output_dir)
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line_5_6] = True



output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir_high_load = os.path.join(tempfile.gettempdir(),"time_series_example", "high_load")
print("Results can be found in your local temp folder: {}".format(output_dir_high_load))
if not os.path.exists(output_dir_high_load):
    os.mkdir(output_dir_high_load)
high_load(net, n_time_steps, output_dir_high_load)

net = simple_test_net()


output_dir_low_load = os.path.join(tempfile.gettempdir(), "time_series_example", "low_load")
print("Results can be found in your local temp folder: {}".format(output_dir_low_load))
if not os.path.exists(output_dir_low_load):
    os.mkdir(output_dir_low_load)
low_load(net, n_time_steps, output_dir_low_load)

net = simple_test_net()


output_dir_gen_disc_high = os.path.join(tempfile.gettempdir(), "time_series_example", "gen_discon_high")
print("Results can be found in your local temp folder: {}".format(output_dir_gen_disc_high))
if not os.path.exists(output_dir_gen_disc_high):
    os.mkdir(output_dir_gen_disc_high)
gen_discon_high(net, n_time_steps, output_dir_gen_disc_high)

net = simple_test_net()

output_dir_gen_disc_low = os.path.join(tempfile.gettempdir(), "time_series_example", "gen_discon_low")
print("Results can be found in your local temp folder: {}".format(output_dir_gen_disc_low))
if not os.path.exists(output_dir_gen_disc_low):
    os.mkdir(output_dir_gen_disc_low)
gen_discon_low(net, n_time_steps, output_dir_gen_disc_low)

net = simple_test_net()

output_dir_line_disc_high = os.path.join(tempfile.gettempdir(), "time_series_example", "line_disc_high")
print("Results can be found in your local temp folder: {}".format(output_dir_line_disc_high))
if not os.path.exists(output_dir_line_disc_high):
    os.mkdir(output_dir_line_disc_high)
line_discon_high(net, n_time_steps, output_dir_line_disc_high)

net = simple_test_net()

output_dir_line_disc_low = os.path.join(tempfile.gettempdir(), "time_series_example", "line_disc_low")
print("Results can be found in your local temp folder: {}".format(output_dir_line_disc_low))
if not os.path.exists(output_dir_line_disc_low):
    os.mkdir(output_dir_line_disc_low)
line_discon_low(net, n_time_steps, output_dir_line_disc_low)



    
    
# Now we want to read the excel files for voltage pu and angle and merge
# The data in a panda file format 
# HIGH LOAD
vpu_high_load_file = os.path.join(output_dir_high_load, "res_bus", "vm_pu.xls")
read_vpu_high_load = pd.read_excel(vpu_high_load_file, index_col=0)

angle_high_load_file = os.path.join(output_dir_high_load, "res_bus", "va_degree.xls")
read_angle_high_load = pd.read_excel(angle_high_load_file, index_col=0)

high_load_df = pd.concat([read_vpu_high_load, read_angle_high_load], axis=1, ignore_index=True)
high_load_df['check_os'] = 'high load'

# LOW LOAD
vpu_low_load_file = os.path.join(output_dir_low_load, "res_bus", "vm_pu.xls")
read_vpu_low_load = pd.read_excel(vpu_low_load_file, index_col=0)

angle_low_load_file = os.path.join(output_dir_low_load, "res_bus", "va_degree.xls")
read_angle_low_load = pd.read_excel(angle_low_load_file, index_col=0)

low_load_df = pd.concat([read_vpu_low_load, read_angle_low_load], axis=1, ignore_index=True)
low_load_df['check_os'] = 'low load'

# GGEN DISCONECTED HIGH
vpu_gen_disc_file_high = os.path.join(output_dir_gen_disc_high, "res_bus", "vm_pu.xls")
read_vpu_gen_disc_high = pd.read_excel(vpu_gen_disc_file_high, index_col=0)

angle_gen_disc_file_high = os.path.join(output_dir_gen_disc_high, "res_bus", "va_degree.xls")
read_angle_gen_disc_high = pd.read_excel(angle_gen_disc_file_high, index_col=0)

gen_disc_high_df = pd.concat([read_vpu_gen_disc_high, read_angle_gen_disc_high], axis=1, ignore_index=True)
gen_disc_high_df['check_os'] = 'Generator disconnected high'

# GGEN DISCONECTED LOW
vpu_gen_disc_file_low = os.path.join(output_dir_gen_disc_low, "res_bus", "vm_pu.xls")
read_vpu_gen_disc_low = pd.read_excel(vpu_gen_disc_file_low, index_col=0)

angle_gen_disc_file_low = os.path.join(output_dir_gen_disc_low, "res_bus", "va_degree.xls")
read_angle_gen_disc_low = pd.read_excel(angle_gen_disc_file_low, index_col=0)

gen_disc_low_df = pd.concat([read_vpu_gen_disc_low, read_angle_gen_disc_low], axis=1, ignore_index=True)
gen_disc_low_df['check_os'] = 'Generator disconnected low'

# LINE DISCONNECTED HIGH LOAD
vpu_line_disc_file_high = os.path.join(output_dir_line_disc_high, "res_bus", "vm_pu.xls")
read_vpu_line_disc_high = pd.read_excel(vpu_line_disc_file_high, index_col=0)


angle_line_disc_file_high = os.path.join(output_dir_line_disc_high, "res_bus", "va_degree.xls")
read_angle_line_disc_high = pd.read_excel(angle_line_disc_file_high, index_col=0)

line_disc_high_df = pd.concat([read_vpu_line_disc_high, read_angle_line_disc_high], axis=1, ignore_index=True)
line_disc_high_df['check_os'] = 'Line disconnected high'

# LINE DISCONNECTED LOW LOAD
vpu_line_disc_file_low = os.path.join(output_dir_line_disc_low, "res_bus", "vm_pu.xls")
read_vpu_line_disc_low = pd.read_excel(vpu_line_disc_file_low, index_col=0)


angle_line_disc_file_low = os.path.join(output_dir_line_disc_low, "res_bus", "va_degree.xls")
read_angle_line_disc_low = pd.read_excel(angle_line_disc_file_low, index_col=0)

line_disc_low_df = pd.concat([read_vpu_line_disc_low, read_angle_line_disc_low], axis=1, ignore_index=True)
line_disc_low_df['check_os'] = 'Line disconnected low'


vdata = pd.concat([high_load_df, low_load_df, gen_disc_high_df, gen_disc_low_df, line_disc_high_df, line_disc_low_df], 
                  axis=0, ignore_index=True)
print(np.shape(vdata))

def plot_simulation_result():
    fig, ax = plt.subplots(nrows=6, figsize=(6, 12))
    # Plotting
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    voltage_df = [read_vpu_high_load, read_vpu_low_load,
                      read_vpu_gen_disc_high, read_vpu_gen_disc_low,
                      read_vpu_line_disc_high, read_vpu_line_disc_low]
    angle_df = [read_angle_high_load, read_angle_low_load,
                    read_angle_gen_disc_high, read_angle_gen_disc_low,
                    read_angle_line_disc_high, read_angle_line_disc_low]
    title_list = ['Base configuration, high load',
                      'Base configuration, low load',
                      'Gen 3 disconnected, high load',
                      'Gen 3 disconnected, low load',
                      'Line 5-6 disconnected, high load',
                      'Line 5-6 disconnected, high load']
    for j in range(0, 6):
            for i in range(0, 9):
                ax[j].scatter(voltage_df[j][i], angle_df[j][i], c=color[i], s=5, label='Bus {}'.format(i + 1))
                box = ax[j].get_position()
                ax[j].set_position([-0.075, box.y0, box.width, box.height])
                ax[j].set_title(title_list[j])
                ax[j].set_xlabel('Voltage')
                ax[j].set_ylabel('Angle')

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',ncol=1, fancybox=True, shadow=True)
    plt.show()
    fig_file_name = 'plot.png'
    fig.savefig(fig_file_name)

plot = plot_simulation_result()
vdataset = vdata.drop(['check_os'], axis=1)
vdataset_for_normalization = vdataset
vdata_normalized = vdataset.copy()
print(vdataset)

for i in range(1, 9):
    vdata_normalized[i] = np.divide(vdataset_for_normalization[i] - vdataset_for_normalization[i].min(),
                                      vdataset_for_normalization[i].max() - vdataset_for_normalization[i].min())
print(vdataset)
for i in range(10, 18):
    vdata_normalized[i] = np.divide(vdataset_for_normalization[i] - vdataset_for_normalization[i].min(),
                                      vdataset_for_normalization[i].max() - vdataset_for_normalization[i].min())

vdataset_norm_labeled = vdata_normalized.copy()
vdataset_norm_labeled['check_os'] = vdata['check_os'].copy()
train_coefficient = 0.8
n_training = int(train_coefficient * n_time_steps)

# Here we separate the training set from the test set

vdataset_train_norm_labeled = pd.concat([vdataset_norm_labeled[:n_training],
                                        vdataset_norm_labeled[n_time_steps:n_training + n_time_steps],
                                        vdataset_norm_labeled[2 * n_time_steps:n_training + 2 * n_time_steps],
                                        vdataset_norm_labeled[3 * n_time_steps:n_training + 3 * n_time_steps],
                                        vdataset_norm_labeled[4 * n_time_steps:n_training + 4 * n_time_steps],
                                        vdataset_norm_labeled[5 * n_time_steps:n_training + 5 * n_time_steps]],
                                        axis=0, ignore_index=True)

vdataset_test_norm_labeled = pd.concat([vdataset_norm_labeled[n_training:n_time_steps],
                                        vdataset_norm_labeled[n_training + n_time_steps:2 * n_time_steps],
                                        vdataset_norm_labeled[n_training + 2 * n_time_steps:3 * n_time_steps],
                                        vdataset_norm_labeled[n_training + 3 * n_time_steps:4 * n_time_steps],
                                        vdataset_norm_labeled[n_training + 4 * n_time_steps:5 * n_time_steps],
                                        vdataset_norm_labeled[n_training + 5 * n_time_steps:6 * n_time_steps]],
                                        axis=0, ignore_index=True)

vdataset.to_excel("vdataset.xlsx")
vdataset_norm_labeled = vdataset_norm_labeled.sample(frac=1).reset_index(drop=True)
vdataset_norm_labeled.to_excel("vdataset_norm_labeled.xlsx")
vdataset_train_norm_labeled = vdataset_train_norm_labeled.sample(frac=1).reset_index(drop=True)
vdataset_test_norm_labeled = vdataset_test_norm_labeled.sample(frac=1).reset_index(drop=True)

Dataset = vdataset.to_numpy()
Dataset_norm = vdata_normalized.to_numpy()



# k-means clustering algorithm

# Choose random centroid first
# np.random.seed(2)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # This is a list of sample indices for each cluster
        # At the beginning of each cluster, there is an empty list
        # Below just sort the indices
        self.clusters = [[] for _ in range(self.K)]
        # Mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialize centroids
        # Replace=False to avoid the same random number?
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        # self.centroids = [self.X[0], self.X[65], self.X[130], self.X[190], self.X[240], self.X[310]]

        # Do optimization
        for _ in range(self.max_iters):
            # Update clusters
            self.clusters = self._create_clusters(self.centroids)
            # if self.plot_steps:
                # self.plot()

            # Update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # if self.plot_steps:
                # self.plot()

            # Check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

        # Return cluster labels
        # self.plot()
        return self._get_cluster_labels(self.clusters)


    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
                # print(cluster_idx)
        return labels

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Initialize centroids
        centroids = np.zeros((self.K, self.n_features))  # this will be tuples
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    # def plot(self):
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     for i, index in enumerate(self.clusters):
    #         point = self.X[index].T
    #         ax.scatter(*point)

    #     for point in self.centroids:
    #         ax.scatter(*point, marker="x", color="black", linewidth=2)
    #     plt.show()


clusterization = KMeans(K=6, max_iters=500, plot_steps=False)
y_pred = clusterization.predict(vdataset_norm_labeled.drop(['check_os'], axis=1).to_numpy())
print(len(y_pred))
# print(type(y_pred[14]))
print(clusterization.clusters)


# Knn nearest neighbors algorithm

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class knn:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_knn (self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # To compute distances beween x and all sampleds in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
            

# create the training set
X_training_set = vdataset_train_norm_labeled.drop(['check_os'], axis=1).to_numpy()
y_training_set = vdataset_train_norm_labeled['check_os'].to_numpy()

# create the testing set
X_testing_set = vdataset_test_norm_labeled.drop(['check_os'], axis=1).to_numpy()
y_testing_set = vdataset_test_norm_labeled['check_os'].to_numpy()

print(X_training_set)
print(y_training_set)
print(X_testing_set)
print(y_testing_set)


clf = knn(k=6)

clf.fit(X_training_set, y_training_set)
prediction = clf.predict_knn(X_testing_set)
print("prediction", prediction)
print("test data", y_testing_set)
print("Accuracy ", accuracy(y_testing_set, prediction))

