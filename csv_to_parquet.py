import pyarrow.csv as pv
import pyarrow.parquet as pq
from pathlib import Path


def csv_to_parquet(data_path: Path) -> None:
    table = pv.read_csv(data_path)
    pq.write_table(table, data_path.with_suffix('.parquet'))