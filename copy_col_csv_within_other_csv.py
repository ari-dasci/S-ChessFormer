import csv
import sys

def copy_column_to_csv(source_csv, target_csv, column_name):
    try:
        # Read the source CSV and extract the specified column
        with open(source_csv, mode='r', encoding='utf-8') as src_file:
            reader = csv.DictReader(src_file)
            if column_name not in reader.fieldnames:
                raise ValueError(f"Column '{column_name}' not found in source CSV.")
            column_data = [row[column_name] for row in reader]

        # Read the target CSV and append the new column
        with open(target_csv, mode='r', encoding='utf-8') as tgt_file:
            reader = csv.DictReader(tgt_file)
            target_fieldnames = reader.fieldnames
            if column_name in target_fieldnames:
                raise ValueError(f"Column '{column_name}' already exists in target CSV.")
            target_data = [row for row in reader]

        # Write the updated target CSV with the new column
        with open(target_csv, mode='w', encoding='utf-8', newline='') as tgt_file:
            writer = csv.DictWriter(tgt_file, fieldnames=target_fieldnames + [column_name])
            writer.writeheader()
            for i, row in enumerate(target_data):
                row[column_name] = column_data[i] if i < len(column_data) else ''
                writer.writerow(row)

        print(f"Column '{column_name}' successfully copied to '{target_csv}'.")

    except Exception as e:
        print(f"Error: {e}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python copy_col_csv_within_other_csv.py <source_csv> <target_csv> <column_name>")
    else:
        source_csv = sys.argv[1]
        target_csv = sys.argv[2]
        column_name = sys.argv[3]
        copy_column_to_csv(source_csv, target_csv, column_name)