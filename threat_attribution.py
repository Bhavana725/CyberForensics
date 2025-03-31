# threat_attribution.py
import streamlit as st
import json
import pandas as pd

def show_attribution_section():
    st.write("### 🕵️ Threat Attribution")
    st.write("Performing threat attribution analysis...")
    try:
        with open("forensic_logs.json", "r") as f:
            logs = json.load(f)  # Load the entire JSON array

        if not logs:
            st.warning("No forensic logs available for threat attribution.")
            return

        # Simple attribution example: count threats by detection type
        threat_counts = {}
        for log in logs:
            if log['threat_level'] == 'high':
                detection_type = log['detection_type']
                threat_counts[detection_type] = threat_counts.get(detection_type, 0) + 1

        if threat_counts:
            st.write("#### Threat Counts by Detection Type:")
            df = pd.DataFrame(list(threat_counts.items()), columns=['Detection Type', 'Threat Count'])
            st.dataframe(df)
        else:
            st.write("No high-level threats detected in the logs.")

    except FileNotFoundError:
        st.warning("No forensic logs found.")
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON format in forensic_logs.json. Please check the file.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
