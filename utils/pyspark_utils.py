from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from typing import Dict, List, Any, Callable
import logging

def select_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    """Selects specified columns from the DataFrame."""
    try:
        return df.select(*columns)
    except Exception as e:
        logging.error(f"Error selecting columns: {e}")
        raise

def filter_rows(df: DataFrame, condition: str) -> DataFrame:
    """Filters the DataFrame based on a given condition."""
    try:
        return df.filter(condition)
    except Exception as e:
        logging.error(f"Error filtering rows: {e}")
        raise

def add_column(df: DataFrame, column_name: str, expression: str) -> DataFrame:
    """Adds a new column to the DataFrame."""
    try:
        return df.withColumn(column_name, F.expr(expression))
    except Exception as e:
        logging.error(f"Error adding column: {e}")
        raise

def rename_column(df: DataFrame, old_name: str, new_name: str) -> DataFrame:
    """Renames a column in the DataFrame."""
    try:
        return df.withColumnRenamed(old_name, new_name)
    except Exception as e:
        logging.error(f"Error renaming column: {e}")
        raise

def aggregate_data(df: DataFrame, group_by: List[str], aggregations: List[List[str]]) -> DataFrame:
    """Aggregates data in the DataFrame."""
    try:
        agg_funcs = {
            "sum": F.sum,
            "avg": F.avg,
            "min": F.min,
            "max": F.max,
            "count": F.count
        }
        group_cols = [F.col(col_name) for col_name in group_by]
        agg_expressions = []
        for agg_col, agg_func_name, new_col in aggregations:
            if agg_func_name in agg_funcs:
                agg_func = agg_funcs[agg_func_name]
                agg_expressions.append(agg_func(F.col(agg_col)).alias(new_col))
            else:
                raise ValueError(f"Unsupported aggregation function: {agg_func_name}")
        if len(group_cols) == 0:
            return df.agg(*agg_expressions)
        return df.groupBy(*group_cols).agg(*agg_expressions)
    except Exception as e:
        logging.error(f"Error aggregating data: {e}")
        raise

def join_dataframes(df1: DataFrame, df2: DataFrame, join_condition: str, join_type: str = "inner") -> DataFrame:
    """Joins two DataFrames."""
    try:
        return df1.join(df2, F.expr(join_condition), join_type)
    except Exception as e:
        logging.error(f"Error joining DataFrames: {e}")
        raise

TRANSFORMATIONS: Dict[str, Callable] = {
    "select": select_columns,
    "filter": filter_rows,
    "withColumn": add_column,
    "rename": rename_column,
    "aggregate": aggregate_data,
    "join": join_dataframes,
}

def fetch_data(df: DataFrame) -> List[Any]:
    """Fetches all data from the DataFrame."""
    try:
        return df.collect()
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        raise

def display_data(df: DataFrame, num_rows: int = 20) -> None:
    """Displays the first few rows of the DataFrame."""
    try:
        df.show(num_rows)
    except Exception as e:
        logging.error(f"Error displaying data: {e}")
        raise

def save_data(df: DataFrame, path: str, format: str , mode: str ) -> None:
    """Saves the DataFrame to a specified path."""
    try:
        df.write.format(format).mode(mode).save(path)
        logging.info(f"DataFrame saved to {path} in {format} format with mode {mode}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def get_dataframe(spark: SparkSession, data_source: Dict) -> DataFrame:
    """Reads a DataFrame from a specified data source."""
    try:
        source_type = data_source.get("type")
        options = data_source.get("options", {})
        if not source_type:
            raise ValueError("Data source type must be specified.")
        if source_type == "csv":
            df = spark.read.csv(options.get("path", ""), header=options.get("header", False),
                                 sep=options.get("delimiter", ","))
        elif source_type == "json":
            df = spark.read.json(options.get("path", ""))
        elif source_type == "parquet":
            df = spark.read.parquet(options.get("path", ""))
        elif source_type == "text":
             df = spark.read.text(options.get("path", ""))
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
        return df
    except Exception as e:
        logging.error(f"Error reading data: {e}")
        raise

ACTIONS: Dict[str, Callable] = {
    "collect": fetch_data,
    "show": display_data,
    "save": save_data,
}

def process_dataframe(spark: SparkSession, config: Dict) -> None:
    total_task_to_complete = config.get("total_task") + 1 
    for i in range(1,total_task_to_complete):
        """Processes a DataFrame based on a configuration dictionary."""
        try:
            source_config = config.get(f"task{i}").get("source")
            if not source_config:
                raise ValueError("Missing 'source' configuration.")
            df = get_dataframe(spark, source_config)
            transformations_config = config.get(f"task{i}").get("transformations", [])
            for transform_config in transformations_config:
                operation = transform_config.get("operation")
                params = transform_config.get("params", {})
                if operation and operation in TRANSFORMATIONS:
                    df = TRANSFORMATIONS[operation](df, **params)
                else:
                    logging.warning(f"Skipping unknown transformation: {operation}")
            action_config = config.get(f"task{i}").get("action")
            if not action_config:
                raise ValueError("Missing 'action' configuration.")
            action_type = action_config.get("type")
            params = action_config.get("params", {})
            if action_type and action_type in ACTIONS:
                df_to_save = df 
                df_to_save.show()  # Show the data
                df_to_save.printSchema() # print the schema
                ACTIONS[action_type](df_to_save, **params)  # Execute the action

            else:
                raise ValueError(f"Unknown action: {action_type}")

        except Exception as e:
            logging.error(f"Error processing DataFrame: {e}")
            raise
