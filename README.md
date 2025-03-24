# **DataProcessor** 
It is a powerful Python-based tool for automating the entire data processing process from uploading to exporting results. The tool is designed to simplify everyday data management tasks without having to write a lot of code.

## Opportunities
- Uploading data from various formats (CSV, Excel, JSON)
- Automatic detection of data types (numeric, categorical)
- Data cleanup:
    - Handling missing values
    - Removing duplicates
    - Detection and treatment of emissions
- Basic data analysis with statistical report generation
- Data visualization:
    - Histograms for numeric fields
    - Pie charts for categorical fields
    - Correlation matrices
- Data development using scikit-learn applications
- Express analysis in various formats (CSV, Excel, JSON)
- Detailed logging of all operations

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-Learn

## Installation
```
git clone https://github.com/whotyc/DataProcessor.git
cd DataProcessor
```
```
pip install -r requirements.txt
```

## Using
### The command line
```
python data_processor.py --input data.csv --output processed_data.csv --clean --visualize
```


## Command Line Options
|  Parameter  |  Reduction  | Description |
|-------------|-------------|-------------|
| --input  | -i  | Path to the input data file   |
| --output   | -o  | Path to save the processed data |
| --format | -f | Output file format (csv, excel, json) |
|--visualize | -v | Create data visualization | 
| --plots-dir | -p | Directory for saving visualization |
| --clean | -c | Perform data cleaning |

## Use as a library
```
from data_processor import DataProcessor

processor = DataProcessor("data.csv")

processor.load_data()

processor.identify_column_types()

analysis = processor.analyze_data()
print(analysis)

clean_data = processor.clean_data()

processor.visualize_data("my_plots")

processor.export_data("processed_data.csv", "csv")
```

## Examples
### Example 1: Basic Data Analysis
```
from data_processor import DataProcessor

processor = DataProcessor("sales_data.csv")
processor.load_data()
processor.identify_column_types()

analysis = processor.analyze_data()
print(f"Dataset size: {analysis['summary']['Data size']}")
print(f"Number of duplicates: {analysis['summary']['Number of duplicates']}")
print(f"Statistics on numeric fields:\n{analysis['numeric_stats']}")
```

## Example 2: Cleaning and visualization
```
from data_processor import DataProcessor

processor = DataProcessor("customer_data.csv")
processor.load_data()
processor.identify_column_types()

clean_data = processor.clean_data(
    handle_missing=True,
    handle_outliers=True,
    remove_duplicates=True
)

processor.visualize_data("customer_insights")

processor.export_data("clean_customer_data.xlsx", "excel")
```

## Example of output visualizations
After running with the --visualize option, the tool will create a set of graphs in the specified directory.:

- Distribution histograms for numeric fields
- Pie charts for categorical fields
- The matrix of correlations between numeric fields
