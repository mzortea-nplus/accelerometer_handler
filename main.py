import yaml
from datetime import datetime
from data_handling import DataOps

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
