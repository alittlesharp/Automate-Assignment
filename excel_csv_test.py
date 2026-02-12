import pandas as pd

def excel_to_csv(input_excel, output_csv):

    try:
        # Read the Excel file into a pandas DataFrame. Only "Sheet1" in cwd.
        df = pd.read_excel(input_excel, sheet_name='Sheet1')
        # Convert the DataFrame to a CSV file
        # index=False prevents pandas from writing the DataFrame index as a column
        # header=True (default) includes the column headers in the CSV
        df.to_csv(output_csv, index=False, header=True)

        print(f"Successfully exported '{input_excel}' to '{output_csv}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_excel}' was not found.")
    except Exception as e:
        print(f"Error: {e}")

input_file = 'jan_tasks.xlsx'
output_file = 'tasks.csv'
excel_to_csv(input_file, output_file)