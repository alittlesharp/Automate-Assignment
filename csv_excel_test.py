import pandas as pd
import os

def csv_to_excel(input_csv, output_excel):
    if not os.path.exists(input_csv):
        print(f"Error: The file '{input_csv}' was not found.")
        return

    try:
        # Read the CSV file into a pandas DataFrame
        # encoding='utf-8' is a common default, but may need adjustment
        df = pd.read_csv(input_csv, encoding='utf-8')

        # Write the DataFrame to an Excel .xlsx file
        df.to_excel(output_excel, index=False, engine='openpyxl')

        print(f"Successfully converted '{input_csv}' to '{output_excel}'.'")

    except UnicodeDecodeError:
        print(f"Error: Could not read the file with utf-8 encoding.")
        print(f"Try a different encoding (e.g., 'latin-1') in pd.read_csv().")
    except Exception as e:
        print(f"An error occurred: {e}")

input_csv = 'new_workload_summary.csv'
output_excel = 'new_workload_summary.xlsx'

csv_to_excel(input_csv, output_excel)