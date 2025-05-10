import sys
import logging
import json
from pyspark.sql import SparkSession
from utils.pyspark_utils import process_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

def main(config_file):
    """
    Main function to load configuration and process DataFrame.
    """
    spark = SparkSession.builder.appName("DynamicDataFrameTransformation") \
        .config("spark.hadoop.fs.permissions.umask-mode", "0022") \
        .config("spark.hadoop.fs.permissions.umask-mode.override", "true") \
        .getOrCreate()

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
    
        process_dataframe(spark, config)
        logging.info("DataFrame processing completed successfully.")
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON: {config_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    """
    Pass config.json as argumnet while running the file.
    Write all the transformations and Actions in config.json file
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file.json>")
        sys.exit(1)
    main(sys.argv[1])