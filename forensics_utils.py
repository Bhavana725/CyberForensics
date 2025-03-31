import json
import pandas as pd
from datetime import datetime

def save_forensic_log(detection_type, results, threat_level, model_type):
    """Enhanced logging function that saves model outputs"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "detection_type": detection_type,
        "model_type": model_type,
        "threat_level": threat_level,
        "results": results.to_dict(orient='records'),
        "metadata": {
            "total_records": len(results),
            "threat_count": sum(results.iloc[:,-1].apply(lambda x: 1 if x in ["Threat", "Block"] else 0)),
            "features_used": list(results.columns[:-1])  # Exclude prediction column
        }
    }
    
    # Append to log file
    with open("forensic_logs.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def load_forensic_data():
    """Load and parse forensic logs"""
    try:
        with open("forensic_logs.json", "r") as f:
            return [json.loads(line) for line in f.readlines()]
    except FileNotFoundError:
        return []
