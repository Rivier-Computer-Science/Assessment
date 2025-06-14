import os
import json
import re
import pandas as pd
from acat import ACAT
import glob

def safe_read_excel(filepath):
    try:
        return pd.read_excel(filepath)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return None

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found - {config_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from config file: {e}")
    return {}

def read_outcomes(outcomes_file):
    df = safe_read_excel(outcomes_file)
    if df is None:
        return []
    df.columns = [col.strip() for col in df.columns]
    co_cols = [col for col in df.columns if col.lower() == 'course outcome']
    if not co_cols:
        print(f"Warning: 'Course Outcome' column not found in {outcomes_file}")
        return []
    outcomes = df[co_cols[0]].dropna().tolist()
    return outcomes

def read_assignments(assignments_file, outcomes):
    df = safe_read_excel(assignments_file)
    if df is None:
        return {}
    df.columns = [str(col).strip() for col in df.columns]
    assignments = {}
    for outcome in outcomes:
        row_matches = df[df.iloc[:, 0] == outcome]
        if not row_matches.empty:
            assignment_names = row_matches.iloc[0, 1:].dropna().tolist()
            assignments[outcome] = assignment_names
    return assignments

def read_grades(grades_file, required_columns):
    df = safe_read_excel(grades_file)
    if df is None:
        return {}
    cleaned_columns = {
        col: re.sub(r"\s*\(.*?\)", "", str(col)).strip()
        for col in df.columns
    }
    df.rename(columns=cleaned_columns, inplace=True)
    if 'SIS User ID' not in df.columns:
        print(f"Error: 'SIS User ID' column missing in grades file {grades_file}")
        return {}
    relevant_columns = ['SIS User ID'] + [col for col in required_columns if col in df.columns]
    df = df[relevant_columns].dropna(subset=['SIS User ID'])
    df.set_index('SIS User ID', inplace=True)
    return df.to_dict(orient='index')

def load_all_cos_from_folder(folder_path):
    co_map = {}
    search_path = os.path.join(folder_path, '*.xlsx')
    files = glob.glob(search_path)
    if not files:
        print(f"Warning: No Excel files found in {folder_path}")
    for filepath in files:
        course_code = os.path.splitext(os.path.basename(filepath))[0]
        df = safe_read_excel(filepath)
        if df is None:
            continue
        df.columns = [str(col).strip() for col in df.columns]
        if 'Course Outcome' in df.columns:
            cos = df['Course Outcome'].dropna().tolist()
            co_map[course_code] = cos
        else:
            print(f"Warning: 'Course Outcome' column not found in {filepath}")
    return co_map

def generate_co_po_mapping(co_map, po_count=12):
    data = []
    for course, cos in co_map.items():
        for co in cos:
            row = [f"{course}: {co}"] + [1 if i % 2 == 0 else 0 for i in range(po_count)]
            data.append(row)
    columns = ['Course Outcome'] + [f"PO{i+1}" for i in range(po_count)]
    return pd.DataFrame(data, columns=columns)

def generate_po_io_mapping(po_count=12, io_count=6):
    data = []
    for i in range(po_count):
        row = [f"PO{i+1}"] + [1 if j % 2 == 0 else 0 for j in range(io_count)]
        data.append(row)
    columns = ['Program Outcome'] + [f"IO{j+1}" for j in range(io_count)]
    return pd.DataFrame(data, columns=columns)

def generate_mappings():
    folder_path = r'course_outcomes\FA24'
    output_folder = 'mappings_output'
    os.makedirs(output_folder, exist_ok=True)
    co_map = load_all_cos_from_folder(folder_path)
    if not co_map:
        print("No Course Outcomes loaded; skipping mapping generation.")
        return
    print(f"Loaded COs for courses: {list(co_map.keys())}")
    co_po_df = generate_co_po_mapping(co_map)
    po_io_df = generate_po_io_mapping()
    co_po_path = os.path.join(output_folder, 'CO_to_PO_Mapping.xlsx')
    po_io_path = os.path.join(output_folder, 'PO_to_IO_Mapping.xlsx')
    try:
        co_po_df.to_excel(co_po_path, index=False)
        po_io_df.to_excel(po_io_path, index=False)
        print(f"Generated CO-to-PO mapping at: {co_po_path}")
        print(co_po_df)
        print(f"Generated PO-to-IO mapping at: {po_io_path}")
        print(po_io_df)
    except Exception as e:
        print(f"Error writing mapping files: {e}")

def main():
    generate_mappings()
    config = load_config("acat_config.json")
    if not config or 'courses' not in config:
        print("Invalid or empty configuration. Exiting.")
        return
    for course in config['courses']:
        course_name = course.get('course_name')
        semester = course.get('semester')
        outcomes_file = course.get('outcomes_file')
        if not course_name or not semester or not outcomes_file:
            print(f"Skipping course due to missing info: {course}")
            continue
        outcomes = read_outcomes(outcomes_file)
        if not outcomes:
            print(f"No outcomes found for course {course_name}, skipping.")
            continue
        print("Processing course:\n", course_name)
        print("Outcomes:\n", outcomes)
        for section_data in course.get('sections', []):
            section = section_data.get('section')
            if not section:
                print("Skipping section with missing section name.")
                continue
            print("\n\nSection", section)
            assignments_mapping = read_assignments(section_data.get('assignments_file', ''), outcomes)
            print("\n\nAssignments Mappings\n", assignments_mapping)
            final_outcomes = {
                outcome: assignments_mapping.get(outcome, [])
                for outcome in outcomes
            }
            print("\n\nFinal Outcomes:\n", final_outcomes)
            required_columns = set(assignment for criteria in final_outcomes.values() for assignment in criteria)
            grades_file = section_data.get('grades_file')
            if not grades_file:
                print(f"Grades file missing for section {section} of course {course_name}, skipping.")
                continue
            student_data = read_grades(grades_file, required_columns=required_columns)
            if not student_data:
                print(f"No student data found for section {section} of course {course_name}, skipping.")
                continue
            print("\n\nread_grades\n", student_data)
            try:
                acat = ACAT(course_name, semester, section, final_outcomes, student_data)
                student_outcomes = acat.compute_course_outcomes()
                acat.summarize_course_outcomes(student_outcomes)
            except Exception as e:
                print(f"Error processing ACAT for course {course_name} section {section}: {e}")
                continue
            excel_output_folder = config.get('output', {}).get('excel_folder')
            database_output_folder = config.get('output', {}).get('database_folder')
            if not excel_output_folder or not database_output_folder:
                print("Output folders not properly configured in config file.")
                return
            excel_output = os.path.join(
                excel_output_folder,
                f"{course_name}_{semester}_{section}_outcomes.xlsx"
            )
            db_output = os.path.join(
                database_output_folder,
                f"{course_name}_{semester}_{section}_outcomes.db"
            )
            os.makedirs(excel_output_folder, exist_ok=True)
            os.makedirs(database_output_folder, exist_ok=True)
            try:
                acat.save_to_excel(student_outcomes, excel_output)
                acat.save_to_sqlite(db_output, student_outcomes)
            except Exception as e:
                print(f"Error saving results for course {course_name} section {section}: {e}")

if __name__ == "__main__":
    main()