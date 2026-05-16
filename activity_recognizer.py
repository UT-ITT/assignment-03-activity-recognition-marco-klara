from pathlib import Path
import numpy as np
import pandas as pd
import time

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

from scipy.signal import find_peaks

from gather_data import sensor, handle_acceleration, handle_gyro, interval, duration

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "assignment-03-training-data-join-this-team-to-upload-your-data"

# helper function to calculate frequency features
def fft_features(signal):
    signal = np.array(signal)

    # Fast fourier transformation
    fft_values = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values)

    # frequency features
    fft_energy_mean = np.mean(fft_magnitude ** 2)

    frequencies = np.fft.rfftfreq(len(signal), d = interval)
    dominant_freq = frequencies[np.argmax(fft_magnitude)]

    spectral_mean = np.mean(fft_magnitude)

    return fft_energy_mean, dominant_freq, spectral_mean

# helper function to calculate signal energy mean
def signal_energy_mean(signal):
    signal = np.array(signal)
    return np.mean(signal ** 2)

# helper function to calculate zero crossing rate
def zero_crossing_rate(signal):
    signal = np.array(signal)
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

# helper function to calculate peak count
def peak_rate(signal):
    peaks, _ = find_peaks(signal)
    return len(peaks) / duration

# only allow certain columns for feature extraction
valid_columns = ("acc_x", "acc_y", "acc_z","gyro_x", "gyro_y", "gyro_z")

# calculate features
def calc_features(df):
    features = {}
    for col in df.columns:
            if (col in valid_columns):
                column = df[col]

                features[f"{col}_mean"] = column.mean() 
                features[f"{col}_std"] = column.std() 
                features[f"{col}_min"] = column.min() 
                features[f"{col}_max"] = column.max() 
                features[f"{col}_median"] = column.median() 
                features[f"{col}_q25"] = column.quantile(0.25) 
                features[f"{col}_q75"] = column.quantile(0.75) 
                features[f"{col}_iqr"] = (column.quantile(0.75) - column.quantile(0.25)) 
                features[f"{col}_skew"] = column.skew() 
                features[f"{col}_energy_mean"] = signal_energy_mean(column) 
                features[f"{col}_zecr"] = zero_crossing_rate(column) 
                features[f"{col}_peak_rate"] = peak_rate(column)  

                energy_fft, dom_freq, spec_mean = fft_features(column)

                features[f"{col}_fft_energy__mean"] = energy_fft
                features[f"{col}_dom_freq"] = dom_freq 
                features[f"{col}_spectral_mean"] = spec_mean 

    return features

# Use windowing to increase the amount of training data and scale down the activity duration
def create_windows(df, window_size, stride):
    windows = []

    for start in range(0, len(df)-window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end]
        windows.append(window)
    return windows

# Calculate features of sample data and store them in a new Dataframe
def csv_feature_extraction(data_directory):
    folder = data_directory

    # collect all csv files inside the directory
    csv_files = list(folder.rglob("*.csv"))

    all_features = []

    # for every file, extract its features (statistical measurements of its column values) 
    # and create a new row in the feature Dataframe
    for file in csv_files:
        global valid_columns

        df = pd.read_csv(file)

        # drop rows that contain NaN / are empty
        df = df.dropna()

        # skip files that dont contain all required columns (valid_columns) or contain zero columns
        skip = False
        for col in valid_columns:
            if col in df.columns:
                if (df[col] == 0).all():
                    skip = True
                    break
            else:
                skip = True
                break
        if skip:
            continue

        # Get activity from filename
        filename_without_suffix = file.stem
        parts = filename_without_suffix.split("-")

        # print error if the file doesnt have the right format i.e. name-activity-number.csv
        if len(parts) != 3:
            print(f"wrong fileformat: {file.name}")
            continue

        activity = parts[1]
        name = parts[0]
        
        # split data into windows
        windows = create_windows(df, 100, 25)

        for window in windows:
            # insert name and activity into a new row
            features = {"name" : name,
                "activity": activity,} 
            
            # append calculated features
            features.update(calc_features(window))
            
            # append this row of features to all features
            all_features.append(features)
    
    # convert list of features into a pandas dataframe
    feature_df = pd.DataFrame(all_features)

    return feature_df


# apply windowing with a window size of 100 (1 second) and stride of 20 (0.2 seconds)
data = csv_feature_extraction(DATA_DIR)

# debug
#print (len(data))
#data.to_csv("debug.csv")
#print(data.columns)
#print(data.head(5))

# Set features and target columns
features = data.drop(columns=["activity", "name"])

target = data["activity"]

#create group train test split to reduce bias
gss = GroupShuffleSplit(test_size=0.2, random_state=42)

names = data["name"]

train_idx, test_idx = next (gss.split(features, target, groups=names))

X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

print(X_test)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

clf.fit(X_train, y_train)

# check prediction accuracies
predictions = clf.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, predictions))
print("F1-score: ", f1_score(y_test, predictions, average = None))

# variables for sensor data
acc_x, acc_y, acc_z = 0, 0, 0
gyro_x, gyro_y, gyro_z = 0, 0, 0

# handle gyroscope and accelerometer data from the DIPPID device
sensor.register_callback('gyroscope', handle_gyro)
sensor.register_callback('accelerometer', handle_acceleration)

# Main loop to constantly evaluate DIPPID data received
def main():
    
    while True:
        activity_data = []
        duration = 2 # seconds
        interval = 0.01 # 100 Hz -> 0.01 seconds per sample
        samples = int(duration/interval)

        # collect activity data 
        for i in range(samples):

            # Collect sensor values
            data_point = {
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z
            }
            activity_data.append(data_point)
            
            # wait for next sample
            time.sleep(interval)

        # list to dataframe
        activity_data_df = pd.DataFrame(activity_data)

        # calculate features (returns dict)
        activity_data_features = calc_features(activity_data_df)

        # predict activity
        current_prediction = clf.predict(pd.DataFrame([activity_data_features]))
        print(current_prediction)


#if __name__ == "__main__":
 #   main()
