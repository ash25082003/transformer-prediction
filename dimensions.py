import pandas as pd
from pathlib import Path
import os
import numpy as np
import ta
import math

class TechnicalIndicators:
    def __init__(self, data, name):
        self.data = data
        self.name = name

    def calculate_ema(self, span, column='Close'):
        self.data[f'EMA_{span}_{column}'] = self.data[column].ewm(span=span, adjust=False).mean().astype(np.float32).round(4)

    def calculate_rsi(self, window):
        self.data[f'RSI_{window}'] = (ta.momentum.rsi(self.data['Close'], window=window).astype(np.float32)/100).round(4)

    def calculate_macd(self):
        self.data['MACD'] = ta.trend.macd(self.data['Close']).astype(np.float32).round(4)
        
    def calculate_fibonacci_pivots(self):
        pivot = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        high = self.data['High']
        low = self.data['Low']
        pivot_range = high - low
        
        # Calculate Fibonacci retracement levels
        R3 = pivot + pivot_range * 1.618
        R2 = pivot + pivot_range * 1.382
        R1 = pivot + pivot_range * 1.236
        S1 = pivot - pivot_range * 1.236
        S2 = pivot - pivot_range * 1.382
        S3 = pivot - pivot_range * 1.618

        # Assign calculated levels to the DataFrame
        self.data['Pivot'] = pivot.astype(np.float32)
        self.data['R1'] = R1.astype(np.float32)
        self.data['R2'] = R2.astype(np.float32)
        self.data['R3'] = R3.astype(np.float32)
        self.data['S1'] = S1.astype(np.float32)
        self.data['S2'] = S2.astype(np.float32)
        self.data['S3'] = S3.astype(np.float32)
        
    def add_date_columns(self):
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data['Day'] = self.data['Date'].dt.day.astype(np.float32)/32
            self.data['Week'] = self.data['Date'].dt.isocalendar().week.astype(np.float32)/52
            self.data['Month'] = self.data['Date'].dt.month.astype(np.float32)/12
            self.data['Weekday'] = self.data['Date'].dt.weekday.astype(np.float32)/8
        else:
            print('No "Date" column found in data.')
            
    def calculate_pct_chng(self):
        self.data['cl_op_t'] = (((self.data['Close'] - self.data['Open']) / self.data['Open'])*100).astype(np.float32)
        self.data['hi_op_t'] = (((self.data['High'] - self.data['Open']) / self.data['Open'])*100).astype(np.float32)
        self.data['lo_op_t'] = (((self.data['Low'] - self.data['Open']) / self.data['Open'])*100).astype(np.float32)
        self.data['op_cl_t_1'] = (((self.data['Open'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1))*100).astype(np.float32)
    
            
    def dropna(self):
        self.data.dropna(axis = 1)
        
    def min_max_volume(self):
        try:
            self.data["Volume"] = (self.data['Volume'] - min(self.data['Volume'])) / (max(self.data['Volume']) - min(self.data['Volume'])) *100
        except:
            print(self.name)
        
    def apply_indicators(self):
        self.add_date_columns()
        self.calculate_pct_chng()
        self.min_max_volume()
        self.dropna()

def process_files(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            name = filename.split("_")[0]
            input_filepath = os.path.join(input_directory, filename)
            output_filepath = os.path.join(output_directory, filename.split("_")[0] + ".csv")
            
            data = pd.read_csv(input_filepath)
            
            # Ensure the data has the necessary 'Close' column
            if 'Close' in data.columns:
                indicators = TechnicalIndicators(data, name)
                indicators.apply_indicators()
                
                data[-2520:].to_csv(output_filepath, index=False)
                print(f'Processed and saved {filename} to {output_directory}')
            else:
                print(f'Skipped {filename} (no "Close" column)')

# Define the directories
input_directory = 'NSE_D'
output_directory = 'dataset/data'

# Process all files in the input directory and save to the output directory
process_files(input_directory, output_directory)
