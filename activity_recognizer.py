from pathlib import Path

import pandas as pd
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler



DATA_DIR = Path("/home/klara/Dokumente/Uni/ITT/woche_4/assignment-03-training-data-join-this-team-to-upload-your-data")

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



data = load_all_csv_data()
modified_data = modify_data(data)
print(f"Loaded {len(data)} rows")
print(modified_data)



