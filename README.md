# Configuration

```yaml
frequency:
  target_fs: 100
  filter_f: 90

fourier:
  nperseg: 4096

date:
  start_date: "2025-03-01"
  end_date: "2025-03-02"

path:
  data_path: "data/12001006920/acc"
  output_path: "data/output/"

questdb:
  addr: "http::addr=192.168.157.242:9000"
```

# Example Usage

Open the configuration file

```python
    with open("src/data_cleaning_config.yaml", "r") as f:
        config = yaml.safe_load(f)
```

Create operator instance

```python
    operator = DataOps()
```

Read data:

- **path**: folder of accelerometer files
- **phm_list**: None to keep all sensors, otherwise specify subset of phms with a list
- **start_date, end_date**: date ranges to filter files

```python
    df = operator.read_files(
        path=config["path"]["data_path"],
        phm_list=None,
        start_date=datetime.strptime(config["date"]["start_date"], "%Y-%m-%d"),
        end_date=datetime.strptime(config["date"]["end_date"], "%Y-%m-%d"),
    )
```

Get segmentated data: splitting is done when holes/gaps between successive measurements are detected (not missing data!!). Typical case is when one hour of measurements is saved every 4 hours.

```python
    df_segments = operator.split_segments(df, min_segment_length_minutes=15)
```

Data pre-processing and cleaning according to the following pipeline:

- **Missing Data**: missing/invalid data - if less than 2 seconds - is interpolated linearly
- **Standardization**: mean removal
- **Denoising**: low-pass filter
- **Downsampling**: for data compression and increased frequency resolution

```python
    for i in range(len(df_segments)):

        df = operator.data_preprocessing(
            df=df_segments[i],
            base_fs=operator.get_avg_fs(df_segments[i]),
            filter_fs=config["frequency"]["filter_f"],
            target_fs=config["frequency"]["target_fs"],
        )
```

Send data to QuestDB

```python
    for i in range(len(df_segments)):
        operator.from_df_to_questdb(
            df=df_segments[i],
            table_name="processed_data",
            addr=config["questdb"]["addr"]
        )
```
