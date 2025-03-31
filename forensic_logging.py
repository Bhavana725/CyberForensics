# forensic_logging.py
import streamlit as st
import json
import pandas as pd
from datetime import datetime

def show_logging_section():
    st.write("### 🪵 Forensic Logging")
    st.write("Displaying forensic logs...")
    try:
        with open("forensic_logs.json", "r") as f:
            logs = json.load(f)  # Load the entire JSON array

        if not logs:
            st.warning("No forensic logs found.")
            return

        # Convert logs to DataFrame for better display
        log_data = []
        for log in logs:
            log_data.append({
                "Timestamp": datetime.fromisoformat(log['timestamp']),
                "Detection Type": log['detection_type'],
                "Threat Level": log['threat_level'],
                "Results": str(log['results'])  # Convert results to string for display
            })

        df = pd.DataFrame(log_data)
        st.dataframe(df)

    except FileNotFoundError:
        st.warning("No forensic logs found.")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format in forensic_logs.json. Please check the file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
