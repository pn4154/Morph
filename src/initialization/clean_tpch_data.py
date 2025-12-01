import os

# Get the project root directory (2 levels up from this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

def clean_tbl_file(file_path):
    print(f"Cleaning {os.path.basename(file_path)}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        line = line.rstrip('\n')
        if line.endswith('|'):
            line = line[:-1]
        cleaned_lines.append(line + '\n')
    
    with open(file_path, 'w') as f:
        f.writelines(cleaned_lines)


def main():
    print(f"Looking for data in: {DATA_DIR}")
    
    tbl_files = [
        'customer.tbl',
        'part.tbl', 
        'orders.tbl',
        'lineitem.tbl'
    ]
    
    for file_name in tbl_files:
        file_path = os.path.join(DATA_DIR, file_name)
        if os.path.exists(file_path):
            clean_tbl_file(file_path)
        else:
            print(f"File not found: {file_path}")


if __name__ == "__main__":
    main()