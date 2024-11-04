import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import ipaddress

class ThreatDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, df):
        # Convert Date and Time
        df['Date'] = pd.to_datetime(df['Date'])
        df['Time'] = pd.to_timedelta(df['Time'])
        df['DateTime'] = df['Date'] + df['Time']
        
        # Sort data
        df = df.sort_values(by=['Target Username', 'DateTime'])
        
        # Feature engineering
        df['encoded_time'] = (df['DateTime'] + pd.Timedelta(hours=1)).dt.floor('H').dt.hour
        df['TimeDiff'] = df.groupby('Target Username')['DateTime'].diff().dt.total_seconds()
        
        # Flag generation
        df['BruteForceFlag'] = (df['TimeDiff'] < 120) & (df['Event ID'] == 4625)
        df['RogueUserFlag'] = ~df['Target Username'].str.match(r'^[A-Za-z_]+$')
        df['UsernameEncoded'] = self.label_encoder.fit_transform(df['Target Username'])
        
        # IP processing
        df['SuspiciousIPFlag'] = ~df['IP'].str.match(r"^192\.168\.\d{1,3}\.\d{1,3}$")
        df['IPInteger'] = df['IP'].apply(self._ip_to_int)
        
        # Additional features
        df['LogonCount'] = df.groupby(['UsernameEncoded', 'Date'])['Event ID'].transform('count')
        df['Hour'] = df['DateTime'].dt.hour
        df['AnomalousTimeFlag'] = (df['Hour'] < 8) | (df['Hour'] > 18)
        
        features = df[['UsernameEncoded', 'Event ID', 'IPInteger', 'LogonCount', 'Hour', 'TimeDiff']].fillna(0)
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features, df
    
    def _ip_to_int(self, ip):
        try:
            return int(ipaddress.ip_address(ip))
        except ValueError:
            return 0
    
    def train(self, X, y):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save_model(self, path):
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)
    
    def load_model(self, path):
        self.model = joblib.load(path)