from pathlib import Path

import pandas as pd
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent / "assignment-03-training-data-join-this-team-to-upload-your-data"

#DATA_DIR = Path("/home/klara/Dokumente/Uni/ITT/woche_4/assignment-03-training-data-join-this-team-to-upload-your-data")

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

def csv_feature_extraction(data_directory):

    folder = data_directory

    # collect all csv files inside the directory
    csv_files = list(folder.rglob("*.csv"))

    all_features = []

    # for every file, extract its features (statistical measurements of its column values) 
    # and create a new row in the feature Dataframe
    for file in csv_files:

        df = pd.read_csv(file)

        # drop rows that contain NaN / are empty
        df = df.dropna()

        # only allow certain columns for feature extraction
        valid_columns = ("acc_x", "acc_y", "acc_z","gyro_x", "gyro_y", "gyro_z")

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

        features = {
            "activity": activity,
        }
        
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

        all_features.append(features)
    
    feature_df = pd.DataFrame(all_features)

    return feature_df

data = csv_feature_extraction(DATA_DIR)
data.to_csv("debug.csv")
print(data.columns)
print(data.head(5))

#data = load_all_csv_data()
#modified_data = modify_data(data)
#print(f"Loaded {len(data)} rows")
#print(modified_data)



