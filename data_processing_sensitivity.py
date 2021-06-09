"""
By Peng Wu, 04/08/2020
Data processing module, based on the Matlab implementation by Enrico Anderlini.
This script:


0. Automatic load all raw data of multiple vehicles with multiple parameters

1. Filters:
    a. Keep only the dive profiles for which there are data from all sensors:
        i. Manually set the sensors not of interests or automatically remove parameters with high empty rates
        ii. Ignore data with empty record of interested measurements
    b. Remove dives where the depth did not reach a depth of 25 m
        i. 25m filtering of depth
        ii. Minimum number of measurement points filtering

2. Resampling
    a. Apply uniformed timeline for all measurements by 1D interpolation

3. Processing for data-driven analysis
    a. Normalisation


# Note:
-   the dictionary 'data' is always the data collection currently under construction in all processing procedures
-   in all processing functions, the input data is not changed, we simply use input data to construct the processed data
-   the data is structured as nested dictionaries:
    data[vehicle][cycle][parameter][records]
    where:
        vehicle is the vehicle name
        cycle is the cycle number
        parameter is the name of the sensor/parameter
        records is the actual measurements
"""


import numpy as np
import os
import pickle
from scipy import interpolate
import time
from copy import deepcopy
from slocum import Slocum
import pandas as pd
import random


class DataProcessingNew:
    def __init__(self, verbose=True, auto_remove=False):
        """
        Locate the data path
        Set function parameters
        Initialise intermediate data
        """
        # set numpy random seed
        np.random.seed(111)

        # Raw data
        self.data_raw = {}

        # filtered data
        self.data_filter = {}

        # interpolated data
        self.data_interp = {}

        # resampled data
        self.data_resample = {}

        # data for model based
        self.data_model = {}

        # data for data-driven
        self.data_deep = {}

        self.path = '/media/peng/Data/PengData/SlocumData/Postprocessed'

        self.verbose = verbose

        # function timer
        self.timer = None
        # time step for interpolation, default 5 s
        self.dt = 5

        # minimum no. of points
        self.np = 9

        # empty rate threshold
        self.e_r_threshold = 0.53

        # TODO: user defined threshold, this should be defined by the user after initial processing
        # currently only 'depth < 25' is applied, also need set < > or = etc.
        self.sensor_threshold = {'depth': 25}

        # max and min values of parameters
        self.minmax = None
        self.minmax_multi = None

        # automatically or manually remove parameters not of interests
        self.auto_remove = auto_remove

        # set parameters not of interests if auto_remove is disabled
        # TODO: this is to be chosen by the user through tick boxes if there is a GUI
        self.not_interested = ['heading_control',
                               'pitch_control',
                               'rudder_angle_control',
                               'vbd_control']

        # store maximum time for each dive
        self.ft = None

        # store raw time lines for each vehicle
        self.t = None

        # automatically detect vehicle names and parameters
        self.sensors = {}
        names = os.listdir(self.path)
        for name in names:
            if os.path.isdir(os.path.join(self.path, name)):
                dives = os.listdir(os.path.join(self.path, name))
                for i in dives:
                    if os.path.isdir(os.path.join(self.path, name, i)):
                        temp = os.listdir(os.path.join(self.path, name, i))
                        for sensor in temp:
                            base = os.path.basename(os.path.join(self.path, name, sensor))
                            base = os.path.splitext(base)[0]
                            self.sensors[base] = {}
                        break
                    else:
                        pass
                break

        # print the parameters in the raw data
        if self.verbose:
            print('User defined raw data path is', self.path)

            print('The vehicles are', names)

            print('The measurements are:')
            for key, value in self.sensors.items():
                print(f'{key}: {value}')
            print('Initialisation done.')

    def load_raw(self):
        """
        Read the raw data in *.out files
        And output the data stored in Python dictionary
        """

        # the dictionary to store raw data
        data_raw = {}
        print('Extracting raw data...')
        self.timer = time.time()
        # get the folders' names relating to each glider:
        names = os.listdir(self.path)
        vehicles = {}

        for name in names:
            if os.path.isdir(os.path.join(self.path, name)):
                vehicles[name] = {}

        for vehicle in vehicles:

            if self.verbose:
                print(vehicle)

            cycle_profiles = {}

            current_dir = os.path.join(self.path, vehicle)
            dirlist = []

            # get the list of all cycles
            cycle_list = os.listdir(current_dir)
            for cycle_sequence in cycle_list:
                if os.path.isdir(os.path.join(current_dir, cycle_sequence)):
                    dirlist.append(cycle_sequence)

            # initialise data, see header about data
            data = {}

            for cycle in dirlist:
                # initialise list of sensors
                sensors = deepcopy(self.sensors)
                for sensor in sensors:
                    data_path = os.path.join(self.path, vehicle, cycle, sensor + '.out')
                    # read the records if it is not empty, otherwise this sensor/parameter is emtpy
                    size = os.stat(data_path).st_size
                    if size:
                        sensors[sensor] = np.genfromtxt(data_path)
                    else:
                        sensors[sensor] = np.array([])

                cycle_profiles[cycle] = deepcopy(sensors)

            data[vehicle] = deepcopy(cycle_profiles)

            # create data folder if there is no
            if not os.path.exists('data'):
                os.makedirs('data')

            # save extracted data to 'data' folder
            pickle.dump(cycle_profiles, open(os.path.join('data', 'rawdata_' + vehicle + '.pkl'), 'wb'))
            if self.verbose:
                print(vehicle, 'raw data loaded & saved to pickle file')

        self.data_raw = deepcopy(data)

        if self.verbose:
            print("--- %f seconds ---" % (time.time() - self.timer))

        return data

    def filter(self):
        """
        This function filters:
        Objective a:
            i. Manually set the sensors not of interests or automatically remove parameters with high empty rates
                -   parameters with high empty rates
                -   parameters not of interests defined by user
            ii. Ignore data with empty record of interested measurements
        Objective b:
            i. 25m filtering of depth,
                TODO: this should be a generic user defined parameter, including the sensors, and their limits
            ii. Minimum number of measurement points filtering
        :return: filtered cycles
        """
        print('Filtering in process...')
        self.timer = time.time()

        if self.auto_remove:
            removed_sensors = self.find_empty_rate()
        else:
            removed_sensors = self.not_interested

        # load raw data
        data_raw = deepcopy(self.data_raw)

        # initialise target data
        data = {}

        # clear memory
        del self.data_raw

        if self.verbose:
            print('Parameters to be removed are:', removed_sensors)

        # remove sensors not of interests
        sensors_filtered = deepcopy(self.sensors)
        for sensor in removed_sensors:
            sensors_filtered.pop(sensor, None)

        for vehicle, cycles in data_raw.items():
            data[vehicle] = {}
            for cycle, sensors in cycles.items():
                data[vehicle][cycle] = {}
                for sensor in sensors_filtered:
                    if sensor in data_raw[vehicle][cycle]:
                        data[vehicle][cycle][sensor] = deepcopy(data_raw[vehicle][cycle][sensor])

        # remove cycles if they contain empty parameters of interest, or the record of any parameter is less than np
        empty = {}
        for vehicle, cycles in data.items():
            empty[vehicle] = []
            for cycle, sensors in cycles.items():
                for sensor, records in sensors.items():
                    if records.shape[0] <= self.np:
                        empty[vehicle].append(cycle)
                        break

        for vehicle in data:
            for cycle in empty[vehicle]:
                del data[vehicle][cycle]
                # print(vehicle, cycle)

        # delete cycles with parameters beyond defined thresholds
        thres = {}
        for vehicle, cycles in data.items():
            thres[vehicle] = []
            for cycle, sensors in cycles.items():
                for r, l in self.sensor_threshold.items():
                    if np.max(sensors[r][:, 1]) < l:
                        thres[vehicle].append(cycle)

        for vehicle in data:
            for cycle in thres[vehicle]:
                del data[vehicle][cycle]

        # remove vehicle if it is empty
        empty_vehicle = []
        for vehicle, cycles in data.items():
            if len(cycles) == 0:
                empty_vehicle.append(vehicle)

        for vehicle in empty_vehicle:
            del data[vehicle]
            if self.verbose:
                print(vehicle, 'is removed due to lack of valuable data: '
                               'try to relax the filtering limits if this vehicle is of interest!')

        """
        to get the raw timelines, including starting and ending time (in raw format), 
        this is to ensure cycles in sequence
        """
        t_pd = {}
        for vehicle, cycles in data.items():
            temp = []
            for cycle, sensors in cycles.items():
                st_raw = np.array([])
                ft_raw = np.array([])
                for sensor, records in sensors.items():
                    st_raw = np.append(st_raw, records[0, 0])
                    ft_raw = np.append(ft_raw, records[-1, 0])

                temp.append(
                    {'cycle': cycle, 's_t': np.min(st_raw), 'f_t': np.max(ft_raw)}
                )

            t_pd[vehicle] = pd.DataFrame(temp)

        t_surface = {}

        for vehicle, temp in t_pd.items():
            t_surface[vehicle] = {}
            temp.sort_values('s_t', inplace=True, ascending=True, ignore_index=False)
            length = temp.shape[0]
            for i in range(length):
                if i == (length-1):
                    t_surface[vehicle][temp.iloc[i, 0]] = 0
                else:
                    t_surface[vehicle][temp.iloc[i, 0]] = temp.iloc[i + 1, 1] - temp.iloc[i, 2]
        self.t = t_surface

        # get vertical velocity

        data = self.get_velocity(data)

        self.data_filter = deepcopy(data)

        if self.verbose:
            print("--- %f seconds ---" % (time.time() - self.timer))

        return data

    def interpolate(self):

        """
        Apply uniformed timeline for all measurements by 1D interpolation
        -   Start time
        -   End time
        -   Relative timelines
        -   Interpolations
        TODO: to include find_delay function included in the Matlab implementation
        :return: data interpolated with uniformed timelines
        """
        print('Interpolation in process...')
        self.timer = time.time()

        data = deepcopy(self.data_filter)

        # set relative timelines for dive profiles with complete records
        ft = {}
        for vehicle, cycles in data.items():
            ft[vehicle] = {}
            for cycle in cycles:
                start_time = np.array([])
                final_time = np.array([])
                # get start and final times of all measurements, find min and max of all measurements for each cycle
                for sensor, measurement in data[vehicle][cycle].items():
                    start_time = np.append(start_time, measurement[0, 0])
                    final_time = np.append(final_time, measurement[-1, 0])

                s_t = np.min(start_time)

                # set relative start time
                for sensor in cycles[cycle]:
                    data[vehicle][cycle][sensor][:, 0] -= s_t

                ft[vehicle][cycle] = np.max(final_time) - s_t

                # apply 1d interpolation to all measurements
                t = np.arange(0, ft[vehicle][cycle], self.dt)

                for sensor, records in cycles[cycle].items():
                    f = interpolate.interp1d(records[:, 0], records[:, 1], fill_value='extrapolate')
                    y = f(t)
                    # clear original data
                    data[vehicle][cycle][sensor] = np.zeros([len(t), 2])
                    # fill in with interpolated data
                    data[vehicle][cycle][sensor][:, 0] = deepcopy(t)
                    data[vehicle][cycle][sensor][:, 1] = deepcopy(y)

        self.ft = ft

        self.data_interp = data

        if self.verbose:
            print("--- %f seconds ---" % (time.time() - self.timer))

        return data

    def model_based_processing(self):

        """
        Processing for model-based analysis:
            a. New parameters introduced and calculation required (Fourth pass, gsw_toolbox called)
            b. Divide the cycles into dives and climbs and remove transients (Fifth pass)
            c. Put together the dives and climbs without transients (Sixth pass or Seventh pass)
            d. Compute the bouncy buoyancy value for each data point (Eighth pass)
        :return: data for system identification
        """
        data_interp = deepcopy(self.data_interp)

        s = Slocum(self.ft, self.dt)

        data_calculated = s.slocum_cal_a(data_interp)

        dive, climb, data = s.slocum_cal_b(data_calculated, self.t)

        data_new = s.slocum_cal_c(dive, climb, data)

        # data for system identification
        data_b = s.slocum_cal_d(data_new)

        return data_b

    def data_driven_processing(self):
        """
        This function process the data for data driven approaches
        TODO: currently only normalisation is included
        :return: normalised data
        """
        print('Prepare data for data-driven approaches...')
        self.timer = time.time()
        data = deepcopy(self.data_interp)

        # normalise the data
        data_normalised_multi = self.normalise_multi(deepcopy(data))

        self.data_multi_train(data_normalised_multi)

        if self.verbose:
            print("--- %f seconds ---" % (time.time() - self.timer))

    def load_extracted(self):
        """
        Load previously extracted
        """
        print('Load pickle files...')
        self.timer = time.time()

        files = os.listdir('data')
        vehicles = []
        data = {}
        for file in files:
            base = os.path.basename(file)
            vehicle = os.path.splitext(base)[0][8:]
            vehicles.append(vehicle)    # remove 'raw_data_' string
            data[vehicle] = pickle.load(open(os.path.join('data', file), 'rb'))
            print(vehicle)

        self.data_raw = deepcopy(data)

        if self.verbose:
            print("--- %f seconds ---" % (time.time() - self.timer))

        return data

    def get_velocity(self, data):
        # replace depth to vertical_velocity for calculation
        for vehicle, cycles in data.items():
            for cycle in cycles:
                data[vehicle][cycle]['vertical_velocity'] = data[vehicle][cycle]['depth'][0:-1, :]

                data[vehicle][cycle]['vertical_velocity'][:, 1] = \
                    np.diff(data[vehicle][cycle]['depth'][:, 1])/np.diff(data[vehicle][cycle]['depth'][:, 0])
        return data

    def find_empty_rate(self):
        """
        This function finds the empty rates of sensors/parameters

        """
        empty_rate = deepcopy(self.sensors)

        for sensor in empty_rate:
            empty_rate[sensor] = 0.

        total_cycle = 0

        data = self.data_raw
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                total_cycle += 1
                for sensor in sensors:
                    size = len(sensors[sensor])
                    if size == 0:
                        empty_rate[sensor] += 1
        for sensor in empty_rate:
            empty_rate[sensor] /= total_cycle

        empty_rate_sorted = sorted(empty_rate.items(), key=lambda x: x[1], reverse=True)

        empty_high = []

        if self.verbose and empty_rate_sorted[0][1] > self.e_r_threshold:
            print('Be careful: the following sensors/parameters have empty rates higher than the threshold of',
                  self.e_r_threshold, '!!!')

        for i in empty_rate_sorted:
            if i[1] > self.e_r_threshold:
                empty_high.append(i[0])
                if self.verbose:
                    print(i[0], ':', i[1])

        print(empty_rate_sorted)
        print('Total cycle number is', total_cycle)

        return empty_high

    def normalise(self, data):
        """
        Normalise all data to the range of [0, 1]
        """
        minmax = {}

        # initialise the normalised data
        for vehicle, cycles in data.items():
            minmax[vehicle] = {}
            for cycle, sensors in cycles.items():
                for sensor in sensors:
                    minmax[vehicle][sensor] = {}
                    minmax[vehicle][sensor]['min'] = None
                    minmax[vehicle][sensor]['max'] = None

        # find min and max for sensors
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                for sensor, records in sensors.items():
                    r = records[:, 1]
                    min_r = r.min()
                    max_r = r.max()
                    if minmax[vehicle][sensor]['min'] is None:
                        minmax[vehicle][sensor]['min'] = min_r
                    else:
                        minmax[vehicle][sensor]['min'] = min(min_r, minmax[vehicle][sensor]['min'])

                    if minmax[vehicle][sensor]['max'] is None:
                        minmax[vehicle][sensor]['max'] = max_r
                    else:
                        minmax[vehicle][sensor]['max'] = max(max_r, minmax[vehicle][sensor]['max'])

        # normalise sensor measurements to [0, 1]
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                for sensor, records in sensors.items():
                    ptp = minmax[vehicle][sensor]['max'] - minmax[vehicle][sensor]['min']
                    data[vehicle][cycle][sensor][:, 1] = (records[:, 1] - minmax[vehicle][sensor]['min']) / ptp
                    # data[vehicle][cycle][sensor][:, 1] = 2 * (records[:, 1] - minmax[vehicle][sensor]['min'])/ptp - 1
        self.minmax = minmax

        # save the minimum and maximum values to file
        if not os.path.exists('training_data'):
            os.makedirs('training_data')

        pickle.dump(minmax, open(os.path.join('training_data', 'de_normal.pkl'), 'wb'))
        return data

    def data_multi_train(self, data):
        """
        This function prepares the data of multiple vehicles of the same type
        -   Training data structure:
            -   Patch: small patches from the processed cycle data
                -   User to define which sensors/parameters not to include
                    -   Currently ignore the location data to ensure transferability
            -   Vehicle: vehicle deployment
            -   Cycle: cycle number/sequence

        -   Generates the:
            -   Training dataset (can also be extended test dataset)
            -   Test dataset (unseen by the model)
            -   Sensitivity study dataset
        """
        no_patches = int(1e5)
        no_steps = 64
        no_patches_test = 10
        # vehicles to be included are as below, N.B. the three Slocum deployments are healthy
        vehicle_list = ['unit_345_d1', 'unit_397_d1']
        print('Deployments for training are:', vehicle_list)
        # remove sensors not to include
        sensor_list = []
        not_to_include = ['gps_lat', 'gps_lon', 'lat', 'lon', 'depth']

        # extract the first XX cycles data
        training_cycles = 400

        # remove vehicles to be used for test
        not_vehicle = []
        for vehicle, cycles in data.items():
            print(len(cycles))
            if vehicle not in vehicle_list:
                not_vehicle.append(vehicle)
                print('Vehicle', vehicle, 'is not included due to insufficient cycles, only', len(cycles), 'cycles')
        print('Deployments for test are:', not_vehicle)

        for vehicle in vehicle_list:
            cycles = data[vehicle]
            for cycle, sensors in cycles.items():
                for sensor in sensors:
                    sensor_list.append(sensor)
                break
            break

        patches_per_vehicle = int(no_patches/len(vehicle_list))
        patches_per_cycle = int(patches_per_vehicle / training_cycles)

        for sensor in not_to_include:
            sensor_list.remove(sensor)

        # create data folder if there is no
        if not os.path.exists('training_data'):
            os.makedirs('training_data')

        # create data folder if there is no
        if not os.path.exists('test_data'):
            os.makedirs('test_data')

        # save sensor list to to training_data folder
        pickle.dump(sensor_list, open(os.path.join('training_data', 'sensor_list_multi' + '.pkl'), 'wb'))
        if self.verbose:
            print('Sensor list for multi vessels saved to pickle file in training_data folder')

        # process multi-vehicle training data

        dataset = {'data': np.zeros([no_patches, len(sensor_list), no_steps]),
                   'vehicle': [None] * no_patches,
                   'cycle': np.zeros([no_patches])}

        c = 0
        for vehicle in vehicle_list:
            cycles = data[vehicle]
            print('Processing training data of', vehicle, len(cycles), 'patches:', patches_per_cycle)

            cycle_list = list(cycles)

            for _ in range(training_cycles):
                cycle = random.choice(cycle_list)
                sensors = cycles[cycle]
                for k in range(patches_per_cycle):
                    patch = np.empty([len(sensor_list), no_steps])
                    start_idx = None
                    for i in range(len(sensor_list)):
                        sensor = sensor_list[i]
                        if start_idx is None:
                            start_idx = np.random.randint(low=0, high=sensors[sensor].shape[0] - no_steps)
                            end_idx = start_idx + no_steps
                        patch[i, :] = sensors[sensor][:, 1][start_idx:end_idx]

                    if c < no_patches:
                        dataset['data'][c, :, :] = patch
                        dataset['cycle'][c] = cycle
                        dataset['vehicle'][c] = vehicle
                    else:
                        break
                    c += 1

        # save the training data to hard drive
        pickle.dump(dataset, open(os.path.join('training_data', 'multi_vehicle.pkl'), 'wb'))
        if self.verbose:
            print('Training data of multi_vehicle saved to pickle file in training_data folder')

        # prepare test data, note this part is modified for sensitivity study
        dts = np.array([5, 10, 30, 60, 120, 240])    # change here to modify the sensitivity study settings

        # remove 'unit_419_d1' as it is the healthy reference
        not_vehicle.remove('unit_399_d2')
        not_vehicle.remove('unit_205')
        # resample, then interpolate, every time work on one sensor and keep other sensors unchanged
        for s in sensor_list:
            for dt in dts:
                data_val = deepcopy(data)
                for vehicle in not_vehicle:
                    cycles = data_val[vehicle]
                    for cycle, sensors in cycles.items():
                        # restore the interpolated data
                        records = data[vehicle][cycle][s]
                        x = records[:, 0]
                        y = records[:, 1]
                        # resample
                        new_x = np.arange(0, records[-1, 0], dt)
                        f = interpolate.interp1d(x, y, fill_value='extrapolate')
                        new_y = f(new_x)
                        ff = interpolate.interp1d(new_x, new_y, fill_value='extrapolate')

                        # interpolate
                        time_line = np.arange(0, self.ft[vehicle][cycle], 5)
                        data_val[vehicle][cycle][s][:, 0] = deepcopy(time_line)
                        data_val[vehicle][cycle][s][:, 1] = ff(time_line)

                test_data_all = {}
                sensitivity_idx = pickle.load(open(os.path.join('test_data', 'sensitivity_idx.pkl'), 'rb'))
                for vehicle in not_vehicle:
                    print('Test data of', vehicle)
                    cycles = data_val[vehicle]
                    test_cycles = len(cycles)
                    no_patches = test_cycles * no_patches_test

                    test_data = {'data': np.zeros([no_patches, len(sensor_list), no_steps]),
                                       'vehicle': [None] * no_patches,
                                       'cycle': np.zeros([no_patches])}
                    c = 0
                    for cycle, sensors in cycles.items():
                        patch = np.empty([len(sensor_list), no_steps])
                        start_idx_list = sensitivity_idx[vehicle][cycle]
                        for i in range(no_patches_test):
                            start_idx = start_idx_list[i]
                            end_idx = start_idx + no_steps
                            for k in range(len(sensor_list)):
                                sensor = sensor_list[k]
                                patch[k, :] = sensors[sensor][:, 1][start_idx:end_idx]
                            if c < no_patches:
                                test_data['data'][c, :, :] = patch
                                test_data['cycle'][c] = cycle
                                test_data['vehicle'][c] = vehicle
                            else:
                                break
                            c += 1

                    test_data_all[vehicle] = deepcopy(test_data)
                # save test data to 'test_data' folder
                file_name = 'test_data_all_' + s + '_' + str(dt) + '.pkl'
                pickle.dump(test_data_all, open(os.path.join('test_data', file_name), 'wb'))
                if self.verbose:
                    print('Test data of multi_vehicle saved to pickle file in training_data folder', file_name)

        test_data_group_b_ = {}
        for dt in dts:
            data_val = deepcopy(data)
            for vehicle in not_vehicle:
                cycles = data_val[vehicle]
                for cycle, sensors in cycles.items():
                    for s, records in sensors.items():
                        # restore the interpolated data
                        records = data[vehicle][cycle][s]
                        x = records[:, 0]
                        y = records[:, 1]
                        # resample
                        new_x = np.arange(0, records[-1, 0], dt)
                        f = interpolate.interp1d(x, y, fill_value='extrapolate')
                        new_y = f(new_x)
                        ff = interpolate.interp1d(new_x, new_y, fill_value='extrapolate')
                        # interpolate
                        time_line = np.arange(0, self.ft[vehicle][cycle], 5)
                        data_val[vehicle][cycle][s][:, 0] = deepcopy(time_line)
                        data_val[vehicle][cycle][s][:, 1] = ff(time_line)
                test_data = {'data': np.zeros([no_patches, len(sensor_list), no_steps]),
                             'vehicle': [None] * no_patches,
                             'cycle': np.zeros([no_patches])}
                c = 0
                for cycle, sensors in cycles.items():
                    patch = np.empty([len(sensor_list), no_steps])
                    start_idx_list = sensitivity_idx[vehicle][cycle]
                    for i in range(no_patches_test):
                        start_idx = start_idx_list[i]
                        end_idx = start_idx + no_steps
                        for k in range(len(sensor_list)):
                            sensor = sensor_list[k]
                            patch[k, :] = sensors[sensor][:, 1][start_idx:end_idx]
                        if c < no_patches:
                            test_data['data'][c, :, :] = patch
                            test_data['cycle'][c] = cycle
                            test_data['vehicle'][c] = vehicle
                        else:
                            break
                        c += 1
                test_data_group_b_[vehicle] = deepcopy(test_data)
            file_name = 'test_data_group_b_' + str(dt) + '.pkl'
            pickle.dump(test_data_group_b_, open(os.path.join('test_data', file_name), 'wb'))
            if self.verbose:
                print('Test data of multi_vehicle saved to pickle file in training_data folder', file_name)

    def normalise_multi(self, data):
        """
        This normalise the dat for all vehicles to multi-vehicle training, all data are to the range of [0, 1]
        """
        minmax = {}

        # initialise the normalised data
        sensor_list = []
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                for sensor in sensors:
                    sensor_list.append(sensor)
                break
            break

        for sensor in sensor_list:
            minmax[sensor] = {}
            minmax[sensor]['min'] = None
            minmax[sensor]['max'] = None

        # find min and max for the sensors across the vehicles
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                for sensor, records in sensors.items():
                    r = records[:, 1]
                    min_r = r.min()
                    max_r = r.max()
                    if minmax[sensor]['min'] is None:
                        minmax[sensor]['min'] = min_r
                    else:
                        minmax[sensor]['min'] = min(min_r, minmax[sensor]['min'])

                    if minmax[sensor]['max'] is None:
                        minmax[sensor]['max'] = max_r
                    else:
                        minmax[sensor]['max'] = max(max_r, minmax[sensor]['max'])

        # normalise sensor measurements to [0, 1]
        for vehicle, cycles in data.items():
            for cycle, sensors in cycles.items():
                for sensor, records in sensors.items():
                    ptp = minmax[sensor]['max'] - minmax[sensor]['min']
                    data[vehicle][cycle][sensor][:, 1] = (records[:, 1] - minmax[sensor]['min']) / ptp
                    # data[cycle][sensor][:, 1] = 2 * (records[:, 1] - minmax[sensor]['min'])/ptp - 1
        self.minmax_multi = minmax

        # save the minimum and maximum values to file
        if not os.path.exists('training_data'):
            os.makedirs('training_data')

        pickle.dump(minmax, open(os.path.join('training_data', 'de_normal_multi.pkl'), 'wb'))
        return data


if __name__ == "__main__":

    dp = DataProcessingNew()
    _ = dp.load_extracted()
    _ = dp.filter()
    _ = dp.interpolate()

    dp.data_driven_processing()
