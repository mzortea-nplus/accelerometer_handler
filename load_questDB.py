import os
import re
from datetime import datetime
import pandas as pd
from questdb.ingress import Sender
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import ShortTimeFFT as STFT
from scipy.fft import fft, fftfreq
import yaml
import sys

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(SRC_DIR))


class DataOps:
    def __init__(self):
        self.df = pd.DataFrame()

    @staticmethod
    def from_df_to_questdb(df, table_name, addr):
        with Sender.from_conf(addr) as sender:
            sender.dataframe(df=df, table_name=table_name, at="time")

    @staticmethod
    def get_avg_fs(df):
        time_deltas = (df["time"].iloc[1:] - df["time"].iloc[:-1]).dt.total_seconds()
        avg_delta = time_deltas.mean()
        fs = 1 / avg_delta
        return fs

    @staticmethod
    def extract_date_from_filename(filename):
        match = re.search(r"\d{4}-\d{2}-\d{2}_\d{2}", filename)
        if match:
            date_str = match.group(0)
            return datetime.strptime(date_str, "%Y-%m-%d_%H")
        else:
            raise ValueError("No valid date found in filename.")

    @staticmethod
    def data_preprocessing(df, fs, filter_fs=None, target_fs=None):
        df = df.sort_values("time")

        t0 = df["time"].iloc[0]
        t1 = df["time"].iloc[-1]

        target_fs = target_fs if target_fs is not None else fs
        target_filter_freq = filter_fs if filter_fs is not None else target_fs / 2 - 1

        # linear interpolation to fill missing data
        df = df.set_index("time").interpolate(method="time").reset_index()

        # ------------------------------------------------------------------
        # 1. Create uniform time grid at original fs
        # ------------------------------------------------------------------
        uniform_time = pd.date_range(
            start=t0,
            end=t1,
            freq=pd.to_timedelta(1 / fs, unit="s"),
        )

        df_uniform = (
            df.set_index("time")
            .reindex(uniform_time)
            .interpolate(method="time")
            .reset_index()
            .rename(columns={"index": "time"})
        )

        arr = df_uniform.drop(columns=["time"]).to_numpy()

        # ------------------------------------------------------------------
        # 2. Remove mean
        # ------------------------------------------------------------------
        arr -= np.mean(arr, axis=0)

        # ------------------------------------------------------------------
        # 3. Low-pass filter
        # ------------------------------------------------------------------
        sos = signal.iirfilter(
            7,
            target_filter_freq,
            btype="lowpass",
            analog=False,
            ftype="butter",
            output="sos",
            fs=fs,
        )
        arr = signal.sosfiltfilt(sos, arr, axis=0)

        # ------------------------------------------------------------------
        # 4. Resample
        # ------------------------------------------------------------------
        n_samples = int(len(arr) * target_fs / fs)
        arr = signal.resample(arr, n_samples, axis=0)

        # ------------------------------------------------------------------
        # 5. Reconstruct correct time axis
        # ------------------------------------------------------------------
        new_time = t0 + pd.to_timedelta(np.arange(n_samples) / target_fs, unit="s")

        out = pd.DataFrame(arr, columns=df_uniform.columns.drop("time"))
        out["time"] = new_time

        print("Data processed with:")
        print("\t - sampling freq:", fs)
        print("\t - filter freq:", config["frequency"]["filter_f"])

        return out

    @staticmethod
    def detect_holes(df, threshold_seconds=2):
        time_diffs = df["time"].diff().dt.total_seconds().fillna(0)
        expected_diff = 1 / (df["time"][1] - df["time"][0]).total_seconds()
        holes = time_diffs[time_diffs > expected_diff + threshold_seconds]
        if holes:
            print(
                f"Detected {len(holes)} holes in the data at times: {df['time'].iloc[holes].tolist()}"
            )
        return holes.index.tolist()

    @staticmethod
    def split_segments(df, min_segment_length_minutes=15):
        gap_indices = DataOps.detect_holes(df, threshold_seconds=2)

        segments = []
        prev_index = 0
        for gap_index in gap_indices:
            segment = df.iloc[prev_index:gap_index]
            if segment["time"].iloc[-1] - segment["time"].iloc[0] > pd.Timedelta(
                minutes=min_segment_length_minutes
            ):
                segments.append(segment)

        return segments

    @staticmethod
    def read_files(
        path,
        phm_list=None,
        start_date=pd.to_datetime("2025-03-01"),
        end_date=pd.to_datetime("2025-03-02"),
    ):
        print("Loading data from:", config["path"]["data_path"])
        full_df = pd.DataFrame()
        for f in os.listdir(path):
            file_path = os.path.join(path, f)

            if not os.path.isfile(file_path) or not f.endswith(".parquet"):
                continue

            file_date = DataOps.extract_date_from_filename(f)
            if not (start_date <= file_date <= end_date):
                continue

            df = pd.read_parquet(file_path)

            df["time"] = pd.to_datetime(df["time"])

            full_df = pd.concat([full_df, df], ignore_index=True, join="outer")
        if phm_list is not None:
            full_df = full_df[["time"] + phm_list]
        df = full_df.sort_values("time").reset_index(drop=True)
        base_fs = DataOps.get_avg_fs(df)
        print("Data loaded with:")
        print("\t - shape:", df.shape)
        print(
            "\t - time range:",
            df["time"].min(),
            "to",
            df["time"].max(),
            "(",
            df.shape[0] / base_fs / 3600,
            "hours )",
        )
        print("\t - n. sensors:", len(df.columns) - 1)
        print("\t - missing values:", df.isna().sum().sum())
        return full_df


#### NOTE: WE SHOULD CHECK FOR MEMORY USAGE WHEN LOADING LARGE DATASETS ####
#### NOTE: WE SHOULD CHECK FOR HOLES IN THE TIME SERIES DATA BEFORE RESAMPLING ####

if __name__ == "__main__":

    with open("src/data_cleaning_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    operator = DataOps()
    df = operator.read_files(
        config["path"]["data_path"],
        phm_list=None,
        start_date=datetime.strptime(config["date"]["start_date"], "%Y-%m-%d"),
        end_date=datetime.strptime(config["date"]["end_date"], "%Y-%m-%d"),
    )

    df_segments = operator.split_segments(df, min_segment_length_minutes=15)

    for i in range(len(df_segments)):

        df = operator.data_preprocessing(
            df=df_segments[i],
            base_fs=operator.get_avg_fs(df_segments[i]),
            filter_fs=config["frequency"]["filter_f"],
            target_fs=config["frequency"]["target_fs"],
        )

        operator.from_df_to_questdb(
            df, table_name="processed_data", addr=config["questdb"]["addr"]
        )
