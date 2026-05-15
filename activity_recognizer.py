from pathlib import Path
import numpy as np
import pandas as pd
import time

from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from scipy.signal import find_peaks

import DIPPID

from gather_data import sensor, handle_acceleration, handle_gyro

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "assignment-03-training-data-join-this-team-to-upload-your-data"

# use UDP
#PORT = 5700
#sensor = DIPPID.SensorUDP(PORT)


def modify_data(df):
    df = df.dropna()
    df = df[(df["gyro_x"] != 0) & (df["gyro_y"] != 0) & (df["gyro_z"] != 0)]
    df = df.drop(["id"], axis=1)
    
    s_scaler = StandardScaler()
    scaled_samples = s_scaler.fit_transform(df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])
    df_mean = df.copy()
    df_mean[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']] = scaled_samples

    m_scalar = MinMaxScaler()
    m_scalar.fit(df_mean[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])

    scaled_samples = m_scalar.transform(df_mean[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']])
    df_normalized = df_mean.copy()
    df_normalized[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']] = scaled_samples

    df_normalized['class'] = 0
    df_normalized.loc[df_normalized['activity'] == 'rowing', 'class'] = 1    
    df_normalized.loc[df_normalized['activity'] == 'jumpingjacks', 'class'] = 2
    df_normalized.loc[df_normalized['activity'] == 'lifting', 'class'] = 3

    return df_normalized


def load_all_csv_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames = []

    for csv_file in data_dir.rglob("*.csv"):
        parts = csv_file.stem.split("-")
        activity = parts[1] if len(parts) >= 3 else None

        df = pd.read_csv(csv_file)

        if "id" in df.columns and "id.1" in df.columns:
            df = df.drop(columns=["id.1"])

        df = df.assign(
            activity=activity,
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)

    return df

# helper function to calculate frequency features
def fft_features(signal):
    signal = np.array(signal)

    # Fast fourier transformation
    fft_values = np.fft.rfft(signal)
    fft_magnitude = np.abs(fft_values)

    # frequency features
    energy = np.sum(fft_magnitude ** 2)
    dominant_freq = np.argmax(fft_magnitude)
    spectral_mean = np.mean(fft_magnitude)

    return energy, dominant_freq, spectral_mean

# helper function to calculate signal energy
def signal_energy(signal):
    signal = np.array(signal)
    return np.sum(signal ** 2)


# helper function to calculate signal entropy
def entropy (signal, bins = 10):
    hist, _ = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

# helper function to calculate zero crossing rate
def zero_crossing_rate(signal):
    signal = np.array(signal)
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

# helper function to calculate peak count
def peak_count(signal):
    peaks, _ = find_peaks(signal)
    return len(peaks)

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
                features[f"{col}_energy"] = signal_energy(column) 
                features[f"{col}_entropy"] = entropy(column)    
                features[f"{col}_zecr"] = zero_crossing_rate(column) 
                features[f"{col}_peaks"] = peak_count(column)  

                energy_f, dom_freq, spec_mean = fft_features(column)

                features[f"{col}_fft_energy"] = energy_f 
                features[f"{col}_dom_freq"] = dom_freq 
                features[f"{col}_spectral_mean"] = spec_mean 

    return features

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
        if len(parts) > 3:
            print(f"wrong fileformat: {file.name}")
            continue

        activity = parts[1]
        name = parts[0]
        
        # insert name and activity into a new row
        features = {"name" : name,
            "activity": activity,} 
        
        # append calculated features
        features.update(calc_features(df))
        
        # append this row of features to all features
        all_features.append(features)
    
    # convert list of features into a pandas dataframe
    feature_df = pd.DataFrame(all_features)

    return feature_df

data = csv_feature_extraction(DATA_DIR)

# debug
#print (len(data))
#data.to_csv("debug.csv")
#print(data.columns)
#print(data.head(5))

# Set features and target columns
features = data.drop(columns=["activity", "name"])

target = data["activity"]

# Create random train test split
#X_train, X_test, y_train, y_test = train_test_split(features, target, 
 #                                                   test_size = 0.1, random_state = 42)

#create group train test split to reduce bias
gss = GroupShuffleSplit(test_size=0.2, random_state=42)

names = data["name"]

train_idx, test_idx = next (gss.split(features, target, groups=names))

X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

# initialize classifier
#clf = TabPFNClassifier.create_default_for_version(
 #   ModelVersion.V2)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

clf.fit(X_train, y_train)

# check prediction accuracies
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))

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


if __name__ == "__main__":
    main()



