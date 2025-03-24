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
            raise ValueError("Путь к файлу не указан")
            
        logger.info(f"Загрузка данных из {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(file_path)
            elif file_extension == '.xlsx' or file_extension == '.xls':
                self.data = pd.read_excel(file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(file_path)
            elif file_extension == '.sql':
                logger.warning("SQL формат требует настройки подключения к БД")
                return
            else:
                logger.error(f"Неподдерживаемый формат файла: {file_extension}")
                return
                
            logger.info(f"Загружено {len(self.data)} строк и {len(self.data.columns)} столбцов")
            return self.data
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {str(e)}")
            raise
            
    def identify_column_types(self):
        if self.data is None:
            logger.error("Данные не загружены")
            return
            
        logger.info("Определение типов столбцов")
        
        self.numeric_features = list(self.data.select_dtypes(include=['int64', 'float64']).columns)
        self.categorical_features = list(self.data.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        logger.info(f"Найдено {len(self.numeric_features)} числовых и {len(self.categorical_features)} категориальных столбцов")
        
    def analyze_data(self):
        if self.data is None:
            logger.error("Данные не загружены")
            return
            
        logger.info("Выполнение базового анализа данных")

        summary = {
            "Размер данных": self.data.shape,
            "Количество дубликатов": self.data.duplicated().sum(),
            "Пропущенные значения": self.data.isnull().sum().to_dict()
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
            logger.error("Данные не загружены")
            return
            
        logger.info("Начало очистки данных")

        self.processed_data = self.data.copy()

        if remove_duplicates:
            initial_rows = len(self.processed_data)
            self.processed_data = self.processed_data.drop_duplicates()
            removed_rows = initial_rows - len(self.processed_data)
            logger.info(f"Удалено {removed_rows} дубликатов")

        if handle_missing:
            for col in self.numeric_features:
                missing_count = self.processed_data[col].isnull().sum()
                if missing_count > 0:
                    median_value = self.processed_data[col].median()
                    self.processed_data[col].fillna(median_value, inplace=True)
                    logger.info(f"Заполнено {missing_count} пропущенных значений в столбце {col} значением {median_value}")
                    
            for col in self.categorical_features:
                missing_count = self.processed_data[col].isnull().sum()
                if missing_count > 0:
                    mode_value = self.processed_data[col].mode()[0]
                    self.processed_data[col].fillna(mode_value, inplace=True)
                    logger.info(f"Заполнено {missing_count} пропущенных значений в столбце {col} значением '{mode_value}'")

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
                        logger.info(f"Обработано {outliers} выбросов в столбце {col}")
                except Exception as e:
                    logger.warning(f"Не удалось обработать выбросы в столбце {col}: {str(e)}")
                    
        logger.info("Очистка данных завершена")
        return self.processed_data
    
    def create_preprocessing_pipeline(self):
        if not self.numeric_features and not self.categorical_features:
            self.identify_column_types()
            
        logger.info("Создание пайплайна предобработки")

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
            logger.error("Данные не загружены")
            return
            
        data_to_visualize = self.processed_data if self.processed_data is not None else self.data
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Создание визуализаций в папке {output_dir}")
        
        for col in self.numeric_features[:5]: 
            plt.figure(figsize=(10, 6))
            sns.histplot(data_to_visualize[col], kde=True)
            plt.title(f'Распределение значений {col}')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/hist_{col}.png')
            plt.close()
            logger.info(f"Создана гистограмма для {col}")
            
        for col in self.categorical_features:
            if data_to_visualize[col].nunique() <= 10: 
                plt.figure(figsize=(10, 6))
                data_to_visualize[col].value_counts().plot.pie(autopct='%1.1f%%')
                plt.title(f'Распределение категорий {col}')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/pie_{col}.png')
                plt.close()
                logger.info(f"Создана круговая диаграмма для {col}")
                
        if len(self.numeric_features) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = data_to_visualize[self.numeric_features].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Матрица корреляций')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_matrix.png')
            plt.close()
            logger.info("Создана матрица корреляций")
            
        logger.info(f"Визуализации созданы и сохранены в папке {output_dir}")
    
    def export_data(self, output_file, format='csv'):
        if self.processed_data is None:
            if self.data is None:
                logger.error("Нет данных для экспорта")
                return
            data_to_export = self.data
        else:
            data_to_export = self.processed_data
            
        logger.info(f"Экспорт данных в формате {format}")
        
        try:
            if format.lower() == 'csv':
                data_to_export.to_csv(output_file, index=False)
            elif format.lower() == 'excel' or format.lower() == 'xlsx':
                data_to_export.to_excel(output_file, index=False)
            elif format.lower() == 'json':
                data_to_export.to_json(output_file, orient='records')
            else:
                logger.error(f"Неподдерживаемый формат экспорта: {format}")
                return
                
            logger.info(f"Данные успешно экспортированы в {output_file}")
        except Exception as e:
            logger.error(f"Ошибка при экспорте данных: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Инструмент автоматизации обработки данных')
    parser.add_argument('--input', '-i', required=True, help='Путь к входному файлу с данными')
    parser.add_argument('--output', '-o', required=True, help='Путь для сохранения обработанных данных')
    parser.add_argument('--format', '-f', default='csv', help='Формат выходного файла (csv, excel, json)')
    parser.add_argument('--visualize', '-v', action='store_true', help='Создать визуализации данных')
    parser.add_argument('--plots-dir', '-p', default='plots', help='Директория для сохранения визуализаций')
    parser.add_argument('--clean', '-c', action='store_true', help='Выполнить очистку данных')
    
    args = parser.parse_args()
    
    try:
        processor = DataProcessor(args.input)

        processor.load_data()

        processor.identify_column_types()

        analysis_results = processor.analyze_data()
        print("\n=== Общая информация о данных ===")
        for key, value in analysis_results["summary"].items():
            print(f"{key}: {value}")
            
        if args.clean:
            processor.clean_data()
            
        if args.visualize:
            processor.visualize_data(args.plots_dir)

        processor.export_data(args.output, args.format)
        
        logger.info("Обработка данных успешно завершена")
        
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        raise

if __name__ == "__main__":
    main()
