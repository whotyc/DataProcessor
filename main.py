import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import argparse
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, input_file=None):
        self.data = None
        self.processed_data = None
        self.input_file = input_file
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, file_path=None):
        if file_path is None:
            file_path = self.input_file
            
        if file_path is None:
            raise ValueError("The file path is not specified")
            
        logger.info(f"Downloading data from {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                self.data = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            elif file_extension == '.sql':
                logger.warning("The SQL format requires setting up a database connection")
                return
            else:
                logger.error(f"Unsupported file format: {file_extension}")
                return
                
            logger.info(f"Uploaded by {len(self.data)} rows and {len(self.data.columns)} columns")
            return self.data
        except Exception as e:
            logger.error(f"Error when uploading data: {str(e)}")
            raise
            
    def identify_column_types(self):
        if self.data is None:
            logger.error("The data has not been uploaded")
            return
            
        logger.info("Defining column types")
        
        self.numeric_features = list(self.data.select_dtypes(include=['int64', 'float64']).columns)
        self.categorical_features = list(self.data.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        logger.info(f"Found {len(self.numeric_features)} numeric and {len(self.categorical_features)} categorical columns")
        
    def analyze_data(self):
        if self.data is None:
            logger.error("The data has not been uploaded")
            return
            
        logger.info("Performing basic data analysis")

        summary = {
            "Data size": self.data.shape,
            "Number of duplicates": self.data.duplicated().sum(),
            "Missing values": self.data.isnull().sum().to_dict()
        }

        numeric_stats = self.data[self.numeric_features].describe() if self.numeric_features else None
  
        categorical_stats = {}
        for col in self.categorical_features:
            categorical_stats[col] = self.data[col].value_counts().to_dict()
            
        return {
            "summary": summary,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats
        }
        
    def clean_data(self, handle_missing=True, handle_outliers=True, remove_duplicates=True):
        if self.data is None:
            logger.error("The data has not been uploaded")
            return
            
        logger.info("Starting data cleanup")

        self.processed_data = self.data.copy()

        if remove_duplicates:
            initial_rows = len(self.processed_data)
            self.processed_data = self.processed_data.drop_duplicates()
            removed_rows = initial_rows - len(self.processed_data)
            logger.info(f"Removed {removed_rows} duplicates")

        if handle_missing:
            for col in self.numeric_features:
                missing_count = self.processed_data[col].isnull().sum()
                if missing_count > 0:
                    median_value = self.processed_data[col].median()
                    self.processed_data[col].fillna(median_value, inplace=True)
                    logger.info(f"The {missing_count} of missing values in the {col} column is filled with the {median_value} value")
                    
            for col in self.categorical_features:
                missing_count = self.processed_data[col].isnull().sum()
                if missing_count > 0:
                    mode_value = self.processed_data[col].mode()[0]
                    self.processed_data[col].fillna(mode_value, inplace=True)
                    logger.info(f"The {missing_count} of missing values in the {col} column is filled with the value '{mode_value}'")

        if handle_outliers and self.numeric_features:
            for col in self.numeric_features:
                try:
                    Q1 = self.processed_data[col].quantile(0.25)
                    Q3 = self.processed_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = ((self.processed_data[col] < lower_bound) | 
                                (self.processed_data[col] > upper_bound)).sum()
                    
                    if outliers > 0:
                        self.processed_data.loc[self.processed_data[col] < lower_bound, col] = lower_bound
                        self.processed_data.loc[self.processed_data[col] > upper_bound, col] = upper_bound
                        logger.info(f"rocessed {outliers} of outliers in the column {col}")
                except Exception as e:
                    logger.warning(f"Couldn't handle outliers in the column {col}: {str(e)}")
                    
        logger.info("Data cleanup is complete")
        return self.processed_data
    
    def create_preprocessing_pipeline(self):
        if not self.numeric_features and not self.categorical_features:
            self.identify_column_types()
            
        logger.info("Creating a preprocessing pipeline")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
            
        return preprocessor
    
    def visualize_data(self, output_dir='plots'):
        if self.data is None:
            logger.error("The data has not been uploaded")
            return
            
        data_to_visualize = self.processed_data if self.processed_data is not None else self.data
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Creating visualizations in a folder {output_dir}")
        
        for col in self.numeric_features[:5]: 
            plt.figure(figsize=(10, 6))
            sns.histplot(data_to_visualize[col], kde=True)
            plt.title(f'Distribution of values {col}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/hist_{col}.png')
            plt.close()
            logger.info(f"A histogram has been created for {col}")
            
        for col in self.categorical_features:
            if data_to_visualize[col].nunique() <= 10: 
                plt.figure(figsize=(10, 6))
                data_to_visualize[col].value_counts().plot.pie(autopct='%1.1f%%')
                plt.title(f'Distribution of categories {col}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/pie_{col}.png')
                plt.close()
                logger.info(f"A pie chart has been created for {col}")
                
        if len(self.numeric_features) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = data_to_visualize[self.numeric_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('The correlation matrix')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_matrix.png')
            plt.close()
            logger.info("A correlation matrix has been created")
            
        logger.info(f"Visualizations are created and saved in a folder {output_dir}")
    
    def export_data(self, output_file, format='csv'):
        if self.processed_data is None:
            if self.data is None:
                logger.error("There is no data to export")
                return
            data_to_export = self.data
        else:
            data_to_export = self.processed_data
            
        logger.info(f"Exporting data in the format {format}")
        
        try:
            if format.lower() == 'csv':
                data_to_export.to_csv(output_file, index=False)
            elif format.lower() == 'excel' or format.lower() == 'xlsx':
                data_to_export.to_excel(output_file, index=False)
            elif format.lower() == 'json':
                data_to_export.to_json(output_file, orient='records')
            else:
                logger.error(f"Unsupported export format: {format}")
                return
                
            logger.info(f"Data has been successfully exported to {output_file}")
        except Exception as e:
            logger.error(f"Error when exporting data: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Data Processing Automation Tool')
    parser.add_argument('--input', '-i', required=True, help='The path to the input data file')
    parser.add_argument('--output', '-o', required=True, help='The path to save the processed data')
    parser.add_argument('--format', '-f', default='csv', help='Output file format (csv, excel, json)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Create Data visualizations')
    parser.add_argument('--plots-dir', '-p', default='plots', help='Directory for saving visualizations')
    parser.add_argument('--clean', '-c', action='store_true', help='Perform data cleanup')
    
    args = parser.parse_args()
    
    try:
        processor = DataProcessor(args.input)

        processor.load_data()

        processor.identify_column_types()

        analysis_results = processor.analyze_data()
        print("\n=== General information about the data ===")
        for key, value in analysis_results["summary"].items():
            print(f"{key}: {value}")
            
        if args.clean:
            processor.clean_data()
            
        if args.visualize:
            processor.visualize_data(args.plots_dir)

        processor.export_data(args.output, args.format)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"An error has occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
