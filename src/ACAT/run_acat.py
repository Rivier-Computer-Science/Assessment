import streamlit as st
import os
import json
import re
import pandas as pd
from acat import ACAT
import glob
from crewai import Agent, Task, Crew
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
from jsonschema import validate, ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Outcome Assessment System", layout="wide", initial_sidebar_state="expanded")

# JSON schema for config validation
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "courses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "course_name": {"type": "string"},
                    "semester": {"type": "string"},
                    "outcomes_file": {"type": "string"},
                    "sections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "assignments_file": {"type": "string"},
                                "grades_file": {"type": "string"}
                            },
                            "required": ["section", "grades_file"]
                        }
                    }
                },
                "required": ["course_name", "semester", "outcomes_file"]
            }
        },
        "output": {
            "type": "object",
            "properties": {
                "excel_folder": {"type": "string"},
                "database_folder": {"type": "string"},
                "co_po_mapping_file": {"type": "string"},
                "po_io_mapping_file": {"type": "string"}
            }
        }
    },
    "required": ["courses"]
}

def safe_read_excel(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        df = pd.read_excel(filepath)
        if df.empty:
            logger.error(f"Excel file {filepath} is empty")
            st.error(f"Error: Excel file {filepath} is empty")
            return None
        return df
    except FileNotFoundError as e:
        logger.error(str(e))
        st.error(str(e))
    except pd.errors.ParserError:
        logger.error(f"Invalid Excel file format: {filepath}")
        st.error(f"Error: Invalid Excel file format - {filepath}")
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        st.error(f"Error reading {filepath}: {e}")
    return None

def validate_config(config):
    try:
        validate(instance=config, schema=CONFIG_SCHEMA)
        return True
    except ValidationError as e:
        logger.error(f"JSON config validation failed: {e.message}")
        st.error(f"Error: Invalid JSON configuration - {e.message}")
        return False

def load_config(config_path):
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as file:
            config = json.load(file)
        if not validate_config(config):
            return {}
        if not config.get('courses'):
            logger.error("No courses found in configuration")
            st.error("Error: No courses found in configuration")
            return {}
        return config
    except FileNotFoundError as e:
        logger.error(str(e))
        st.error(str(e))
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from config file: {e}")
        st.error(f"Error decoding JSON from config file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        st.error(f"Unexpected error loading config: {e}")
    return {}

def read_outcomes(outcomes_file):
    if not outcomes_file.endswith('.xlsx'):
        logger.error(f"Invalid file format for outcomes: {outcomes_file}. Expected .xlsx")
        st.error(f"Error: Invalid file format for {outcomes_file}. Please use .xlsx")
        return []
    df = safe_read_excel(outcomes_file)
    if df is None:
        return []
    df.columns = [col.strip() for col in df.columns]
    co_cols = [col for col in df.columns if col.lower() == 'course outcome']
    if not co_cols:
        logger.warning(f"'Course Outcome' column not found in {outcomes_file}")
        st.warning(f"Warning: 'Course Outcome' column not found in {outcomes_file}")
        return []
    outcomes = df[co_cols[0]].dropna().tolist()
    if not outcomes:
        logger.warning(f"No valid outcomes found in {outcomes_file}")
        st.warning(f"Warning: No valid outcomes found in {outcomes_file}")
    return outcomes

def read_assignments(assignments_file, outcomes):
    if not assignments_file.endswith('.xlsx'):
        logger.error(f"Invalid file format for assignments: {assignments_file}. Expected .xlsx")
        st.error(f"Error: Invalid file format for {assignments_file}. Please use .xlsx")
        return {}
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
        else:
            logger.warning(f"No assignments found for outcome {outcome} in {assignments_file}")
            st.warning(f"Warning: No assignments found for outcome {outcome}")
    if not assignments:
        logger.warning(f"No valid assignments found in {assignments_file}")
        st.warning(f"Warning: No valid assignments found in {assignments_file}")
    return assignments

def read_grades(grades_file, required_columns):
    if not grades_file.endswith('.xlsx'):
        logger.error(f"Invalid file format for grades: {grades_file}. Expected .xlsx")
        st.error(f"Error: Invalid file format for {grades_file}. Please use .xlsx")
        return {}
    df = safe_read_excel(grades_file)
    if df is None:
        return {}
    cleaned_columns = {
        col: re.sub(r"\s*\(.*?\)", "", str(col)).strip()
        for col in df.columns
    }
    df.rename(columns=cleaned_columns, inplace=True)
    if 'SIS User ID' not in df.columns:
        logger.error(f"'SIS User ID' column missing in grades file {grades_file}")
        st.error(f"Error: 'SIS User ID' column missing in grades file {grades_file}")
        return {}
    relevant_columns = ['SIS User ID'] + [col for col in required_columns if col in df.columns]
    if len(relevant_columns) <= 1:
        logger.warning(f"No required columns found in grades file {grades_file}")
        st.warning(f"Warning: No required columns found in grades file {grades_file}")
        return {}
    df = df[relevant_columns].dropna(subset=['SIS User ID'])
    if df.empty:
        logger.warning(f"No valid data after filtering in grades file {grades_file}")
        st.warning(f"Warning: No valid data after filtering in grades file {grades_file}")
        return {}
    df.set_index('SIS User ID', inplace=True)
    return df.to_dict(orient='index')

def load_all_cos_from_folder(folder_path):
    co_map = {}
    if not os.path.exists(folder_path):
        logger.error(f"Course outcomes folder not found: {folder_path}")
        st.error(f"Error: Course outcomes folder not found: {folder_path}")
        return co_map
    search_path = os.path.join(folder_path, '*.xlsx')
    files = glob.glob(search_path)
    if not files:
        logger.warning(f"No Excel files found in {folder_path}")
        st.warning(f"Warning: No Excel files found in {folder_path}")
    for filepath in files:
        if not filepath.endswith('.xlsx'):
            logger.warning(f"Skipping invalid file format: {filepath}")
            st.warning(f"Warning: Skipping invalid file format: {filepath}")
            continue
        course_code = os.path.splitext(os.path.basename(filepath))[0]
        df = safe_read_excel(filepath)
        if df is None:
            continue
        df.columns = [str(col).strip() for col in df.columns]
        if 'Course Outcome' in df.columns:
            cos = df['Course Outcome'].dropna().tolist()
            if cos:
                co_map[course_code] = cos
            else:
                logger.warning(f"No valid Course Outcomes in {filepath}")
                st.warning(f"Warning: No valid Course Outcomes in {filepath}")
        else:
            logger.warning(f"'Course Outcome' column not found in {filepath}")
            st.warning(f"Warning: 'Course Outcome' column not found in {filepath}")
    return co_map

def generate_co_po_mapping(co_map, po_count=12):
    data = []
    for course, cos in co_map.items():
        for co in cos:
            weights = [1 if i % 2 == 0 else 0 for i in range(po_count)]
            normalized_weights = [w / sum(weights) if sum(weights) > 0 else 0 for w in weights]
            row = [f"{course}: {co}"] + normalized_weights
            data.append(row)
    columns = ['Course Outcome'] + [f"PO{i+1}" for i in range(po_count)]
    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        logger.warning("No CO-to-PO mappings generated")
        st.warning("Warning: No CO-to-PO mappings generated")
    return df

def generate_po_io_mapping(po_count=12, io_count=6):
    data = []
    for i in range(po_count):
        weights = [1 if j % 2 == 0 else 0 for j in range(io_count)]
        normalized_weights = [w / sum(weights) if sum(weights) > 0 else 0 for w in weights]
        row = [f"PO{i+1}"] + normalized_weights
        data.append(row)
    columns = ['Program Outcome'] + [f"IO{j+1}" for j in range(io_count)]
    df = pd.DataFrame(data, columns=columns)
    if df.empty:
        logger.warning("No PO-to-IO mappings generated")
        st.warning("Warning: No PO-to-IO mappings generated")
    return df

def generate_mappings():
    folder_path = r'course_outcomes\FA24'
    output_folder = 'mappings_output'
    os.makedirs(output_folder, exist_ok=True)
    co_map = load_all_cos_from_folder(folder_path)
    if not co_map:
        logger.warning("No Course Outcomes loaded; skipping mapping generation.")
        st.warning("No Course Outcomes loaded; skipping mapping generation.")
        return
    logger.info(f"Loaded COs for courses: {list(co_map.keys())}")
    st.info(f"Loaded COs for courses: {list(co_map.keys())}")
    co_po_df = generate_co_po_mapping(co_map)
    po_io_df = generate_po_io_mapping()
    co_po_path = os.path.join(output_folder, 'CO_to_PO_Mapping.xlsx')
    po_io_path = os.path.join(output_folder, 'PO_to_IO_Mapping.xlsx')
    try:
        co_po_df.to_excel(co_po_path, index=False)
        po_io_df.to_excel(po_io_path, index=False)
        logger.info(f"Generated CO-to-PO mapping at: {co_po_path}")
        st.success(f"Generated CO-to-PO mapping at: {co_po_path}")
        st.dataframe(co_po_df)
        logger.info(f"Generated PO-to-IO mapping at: {po_io_path}")
        st.success(f"Generated PO-to-IO mapping at: {po_io_path}")
        st.dataframe(po_io_df)
    except Exception as e:
        logger.error(f"Error writing mapping files: {e}")
        st.error(f"Error writing mapping files: {e}")

def compute_program_outcomes(config, course_name, semester, section, co_excel_file, output_folder):
    co_df = safe_read_excel(co_excel_file)
    if co_df is None:
        logger.error(f"Could not read CO Excel file {co_excel_file}")
        st.error(f"Error: Could not read CO Excel file {co_excel_file}")
        return
    if 'SIS User ID' not in co_df.columns or 'Course Outcome' not in co_df.columns:
        logger.error(f"Missing required columns in {co_excel_file}")
        st.error(f"Error: Missing required columns in {co_excel_file}")
        return
    student_co_scores = co_df[co_df['SIS User ID'] != 'Class Average'].set_index('SIS User ID').filter(like='CO').dropna()
    if student_co_scores.empty:
        logger.error(f"No valid CO scores for {course_name}_{semester}_{section}")
        st.error(f"Error: No valid CO scores for {course_name}_{semester}_{section}")
        return
    class_co_avg = student_co_scores.mean().to_dict()
    co_po_mapping_file = config.get('output', {}).get('co_po_mapping_file')
    if not co_po_mapping_file:
        logger.error("CO-to-PO mapping file not specified in config")
        st.error("Error: CO-to-PO mapping file not specified in config")
        return
    co_po_df = safe_read_excel(co_po_mapping_file)
    if co_po_df is None:
        logger.error(f"Could not read CO-to-PO mapping file {co_po_mapping_file}")
        st.error(f"Error: Could not read CO-to-PO mapping file {co_po_mapping_file}")
        return
    if 'Course Outcome' not in co_po_df.columns:
        logger.error(f"'Course Outcome' column missing in {co_po_mapping_file}")
        st.error(f"Error: 'Course Outcome' column missing in {co_po_mapping_file}")
        return
    po_columns = [col for col in co_po_df.columns if col.startswith('PO')]
    if not po_columns:
        logger.error(f"No PO columns found in {co_po_mapping_file}")
        st.error(f"Error: No PO columns found in {co_po_mapping_file}")
        return
    course_co_prefix = f"{course_name}: "
    co_po_df = co_po_df[co_po_df['Course Outcome'].str.startswith(course_co_prefix)]
    if co_po_df.empty:
        logger.warning(f"No CO-to-PO mappings found for course {course_name}")
        st.warning(f"Warning: No CO-to-PO mappings found for course {course_name}")
        return
    student_po_scores = {sid: {po: 0.0 for po in po_columns} for sid in student_co_scores.index}
    class_po_scores = {po: 0.0 for po in po_columns}
    co_counts = {po: 0 for po in po_columns}
    weight_sums = {po: 0.0 for po in po_columns}
    for _, row in co_po_df.iterrows():
        co = row['Course Outcome'].replace(course_co_prefix, '')
        if co not in student_co_scores.columns:
            logger.warning(f"CO {co} not found in CO scores for {course_name}")
            st.warning(f"Warning: CO {co} not found in CO scores for {course_name}")
            continue
        for po in po_columns:
            weight = row[po]
            if not isinstance(weight, (int, float)) or weight < 0:
                logger.warning(f"Invalid weight {weight} for PO {po} in CO {co}")
                st.warning(f"Warning: Invalid weight for PO {po} in CO {co}")
                continue
            if weight > 0:
                weight_sums[po] += weight
                for sid in student_co_scores.index:
                    co_score = student_co_scores.at[sid, co]
                    if pd.notna(co_score):
                        student_po_scores[sid][po] += co_score * weight
                class_co_score = class_co_avg.get(co, 0)
                class_po_scores[po] += class_co_score * weight
                co_counts[po] += 1
    for po in po_columns:
        if co_counts[po] > 0 and weight_sums[po] > 0:
            class_po_scores[po] /= weight_sums[po]
            for sid in student_po_scores:
                student_po_scores[sid][po] /= weight_sums[po]
    student_po_df = pd.DataFrame(student_po_scores).T.reset_index().rename(columns={'index': 'SIS User ID'})
    class_po_df = pd.DataFrame([class_po_scores], index=['Class Average'])
    output_df = pd.concat([student_po_df, class_po_df.reset_index().rename(columns={'index': 'SIS User ID'})])
    output_df = output_df.round(2)
    po_output_file = os.path.join(
        output_folder,
        f"{course_name}_{semester}_{section}_po_outcomes.xlsx"
    )
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_df.to_excel(po_output_file, index=False)
        logger.info(f"Saved PO outcomes to {po_output_file}")
        st.success(f"Saved PO outcomes to {po_output_file}")
    except Exception as e:
        logger.error(f"Error saving PO outcomes to {po_output_file}: {e}")
        st.error(f"Error saving PO outcomes to {po_output_file}: {e}")
    return po_output_file

def compute_institutional_outcomes(config, course_name, semester, section, po_excel_file, output_folder):
    po_df = safe_read_excel(po_excel_file)
    if po_df is None:
        logger.error(f"Could not read PO Excel file {po_excel_file}")
        st.error(f"Error: Could not read PO Excel file {po_excel_file}")
        return
    if 'SIS User ID' not in po_df.columns:
        logger.error(f"Missing 'SIS User ID' column in {po_excel_file}")
        st.error(f"Error: Missing 'SIS User ID' column in {po_excel_file}")
        return
    student_po_scores = po_df[po_df['SIS User ID'] != 'Class Average'].set_index('SIS User ID').filter(like='PO').dropna()
    if student_po_scores.empty:
        logger.error(f"No valid PO scores for {course_name}_{semester}_{section}")
        st.error(f"Error: No valid PO scores for {course_name}_{semester}_{section}")
        return
    class_po_avg = po_df[po_df['SIS User ID'] == 'Class Average'].filter(like='PO').iloc[0].to_dict()
    po_io_mapping_file = config.get('output', {}).get('po_io_mapping_file')
    if not po_io_mapping_file:
        logger.error("PO-to-IO mapping file not specified in config")
        st.error("Error: PO-to-IO mapping file not specified in config")
        return
    po_io_df = safe_read_excel(po_io_mapping_file)
    if po_io_df is None:
        logger.error(f"Could not read PO-to-IO mapping file {po_io_mapping_file}")
        st.error(f"Error: Could not read PO-to-IO mapping file {po_io_mapping_file}")
        return
    if 'Program Outcome' not in po_io_df.columns:
        logger.error(f"'Program Outcome' column missing in {po_io_mapping_file}")
        st.error(f"Error: 'Program Outcome' column missing in {po_io_mapping_file}")
        return
    io_columns = [col for col in po_io_df.columns if col.startswith('IO')]
    if not io_columns:
        logger.error(f"No IO columns found in {po_io_mapping_file}")
        st.error(f"Error: No IO columns found in {po_io_mapping_file}")
        return
    student_io_scores = {sid: {io: 0.0 for io in io_columns} for sid in student_po_scores.index}
    class_io_scores = {io: 0.0 for io in io_columns}
    po_counts = {io: 0 for io in io_columns}
    weight_sums = {io: 0.0 for io in io_columns}
    for _, row in po_io_df.iterrows():
        po = row['Program Outcome']
        if po not in student_po_scores.columns:
            logger.warning(f"PO {po} not found in PO scores for {course_name}")
            st.warning(f"Warning: PO {po} not found in PO scores for {course_name}")
            continue
        for io in io_columns:
            weight = row[io]
            if not isinstance(weight, (int, float)) or weight < 0:
                logger.warning(f"Invalid weight {weight} for IO {io} in PO {po}")
                st.warning(f"Warning: Invalid weight for IO {io} in PO {po}")
                continue
            if weight > 0:
                weight_sums[io] += weight
                for sid in student_po_scores.index:
                    po_score = student_po_scores.at[sid, po]
                    if pd.notna(po_score):
                        student_io_scores[sid][io] += po_score * weight
                class_po_score = class_po_avg.get(po, 0)
                class_io_scores[io] += class_po_score * weight
                po_counts[io] += 1
    for io in io_columns:
        if po_counts[io] > 0 and weight_sums[io] > 0:
            class_io_scores[io] /= weight_sums[io]
            for sid in student_io_scores:
                student_io_scores[sid][io] /= weight_sums[io]
    student_io_df = pd.DataFrame(student_io_scores).T.reset_index().rename(columns={'index': 'SIS User ID'})
    class_io_df = pd.DataFrame([class_io_scores], index=['Class Average'])
    output_df = pd.concat([student_io_df, class_io_df.reset_index().rename(columns={'index': 'SIS User ID'})])
    output_df = output_df.round(2)
    io_output_file = os.path.join(
        output_folder,
        f"{course_name}_{semester}_{section}_io_outcomes.xlsx"
    )
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_df.to_excel(io_output_file, index=False)
        logger.info(f"Saved IO outcomes to {io_output_file}")
        st.success(f"Saved IO outcomes to {io_output_file}")
    except Exception as e:
        logger.error(f"Error saving IO outcomes to {io_output_file}: {e}")
        st.error(f"Error saving IO outcomes to {io_output_file}: {e}")
    return io_output_file

def compute_student_assessments(config, course_name, semester, section, co_excel_file, po_excel_file, io_excel_file, output_folder):
    co_df = safe_read_excel(co_excel_file)
    po_df = safe_read_excel(po_excel_file)
    io_df = safe_read_excel(io_excel_file)
    if co_df is None or po_df is None or io_df is None:
        logger.error(f"Could not read input files for student assessments in {course_name}_{semester}_{section}")
        st.error(f"Error: Could not read input files for student assessments in {course_name}_{semester}_{section}")
        return
    student_co_scores = co_df[co_df['SIS User ID'] != 'Class Average'].set_index('SIS User ID').filter(like='CO')
    student_po_scores = po_df[po_df['SIS User ID'] != 'Class Average'].set_index('SIS User ID').filter(like='PO')
    student_io_scores = io_df[io_df['SIS User ID'] != 'Class Average'].set_index('SIS User ID').filter(like='IO')
    if student_co_scores.empty or student_po_scores.empty or student_io_scores.empty:
        logger.error(f"Empty data for CO, PO, or IO in {course_name}_{semester}_{section}")
        st.error(f"Error: Empty data for CO, PO, or IO in {course_name}_{semester}_{section}")
        return
    student_ids = student_co_scores.index.intersection(student_po_scores.index).intersection(student_io_scores.index)
    if student_ids.empty:
        logger.error(f"No common student IDs found for {course_name}_{semester}_{section}")
        st.error(f"Error: No common student IDs found for {course_name}_{semester}_{section}")
        return
    course_outcome_agent = Agent(
        role='Course Outcome Assessment Agent',
        goal='Analyze student performance at the course outcome level and identify strengths and weaknesses.',
        backstory='Expert in evaluating course-level student performance data.'
    )
    program_outcome_agent = Agent(
        role='Program Outcome Assessment Agent',
        goal='Assess student capabilities at the program outcome level.',
        backstory='Specialist in program-level educational assessment.'
    )
    institutional_outcome_agent = Agent(
        role='Institutional Outcome Assessment Agent',
        goal='Evaluate student attainment of institutional goals based on institutional outcome data.',
        backstory='Experienced in institutional-level outcome analysis with a focus on broad educational goals.'
    )
    overall_assessment_agent = Agent(
        role='Student Learning Overall Assessment Agent',
        goal='Combine CO, PO, and IO data to provide comprehensive student capability insights.',
        backstory='Expert in synthesizing multi-level educational data for holistic student assessment.'
    )
    assessments = []
    for sid in student_ids:
        co_data = student_co_scores.loc[sid].to_dict()
        po_data = student_po_scores.loc[sid].to_dict()
        io_data = student_io_scores.loc[sid].to_dict()
        co_task = Task(
            description=f"Analyze course outcome data for student {sid}: {co_data}",
            agent=course_outcome_agent,
            expected_output=f"Textual summary of student {sid}'s strengths and weaknesses in course outcomes."
        )
        po_task = Task(
            description=f"Analyze program outcome data for student {sid}: {po_data}",
            agent=program_outcome_agent,
            expected_output=f"Textual summary of student {sid}'s program-level capabilities."
        )
        io_task = Task(
            description=f"Analyze institutional outcome data for student {sid}: {io_data}",
            agent=institutional_outcome_agent,
            expected_output=f"Textual summary of student {sid}'s attainment of institutional goals."
        )
        overall_task = Task(
            description=f"Combine CO ({co_data}), PO ({po_data}), and IO ({io_data}) data for student {sid} to provide overall capability insights.",
            agent=overall_assessment_agent,
            expected_output=f"Comprehensive textual summary of student {sid}'s overall learning capabilities."
        )
        crew = Crew(
            agents=[course_outcome_agent, program_outcome_agent, institutional_outcome_agent, overall_assessment_agent],
            tasks=[co_task, po_task, io_task, overall_task],
            verbose=False
        )
        try:
            results = crew.kickoff()
            assessment_summary = {
                'SIS User ID': sid,
                'Course Outcome Assessment': results[0] if results else 'No assessment generated',
                'Program Outcome Assessment': results[1] if len(results) > 1 else 'No assessment generated',
                'Institutional Outcome Assessment': results[2] if len(results) > 2 else 'No assessment generated',
                'Overall Assessment': results[3] if len(results) > 3 else 'No assessment generated'
            }
            assessments.append(assessment_summary)
        except Exception as e:
            logger.error(f"Error processing assessments for student {sid}: {e}")
            st.warning(f"Warning: Failed to generate assessments for student {sid}")
    if not assessments:
        logger.error(f"No assessments generated for {course_name}_{semester}_{section}")
        st.error(f"Error: No assessments generated for {course_name}_{semester}_{section}")
        return
    assessment_df = pd.DataFrame(assessments)
    output_file = os.path.join(
        output_folder,
        f"{course_name}_{semester}_{section}_student_assessment.xlsx"
    )
    try:
        os.makedirs(output_folder, exist_ok=True)
        assessment_df.to_excel(output_file, index=False)
        logger.info(f"Saved student assessments to {output_file}")
        st.success(f"Saved student assessments to {output_file}")
    except Exception as e:
        logger.error(f"Error saving student assessments to {output_file}: {e}")
        st.error(f"Error saving student assessments to {output_file}: {e}")

def generate_comparison_charts(dfs, tab_name, course_filter, section_filter, semester_filter, outcome_filter, score_range, group_by):
    if not dfs:
        logger.warning("No data available for comparison")
        st.warning("No data available for comparison.")
        return
    combined_df = pd.concat([df.assign(Source=os.path.basename(file)) for file, df in dfs.items()])
    if 'SIS User ID' in combined_df.columns:
        combined_df = combined_df[combined_df['SIS User ID'] != 'Class Average']
    if course_filter != "All":
        combined_df = combined_df[combined_df['Source'].str.startswith(course_filter)]
    if section_filter != "All":
        combined_df = combined_df[combined_df['Source'].str.contains(section_filter)]
    if semester_filter != "All":
        combined_df = combined_df[combined_df['Source'].str.contains(semester_filter)]
    if score_range:
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            combined_df = combined_df[(combined_df[col] >= score_range[0]) & (combined_df[col] <= score_range[1])]
    if outcome_filter and outcome_filter != "All":
        if outcome_filter in combined_df.columns:
            combined_df = combined_df[['SIS User ID', outcome_filter, 'Source']]
        else:
            logger.warning(f"Outcome {outcome_filter} not found in data")
            st.warning(f"Outcome {outcome_filter} not found in data.")
            return
    if combined_df.empty:
        logger.warning("No data available after applying filters")
        st.warning("No data available after applying filters.")
        return
    if group_by == "Student":
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            fig = go.Figure()
            for sid in combined_df['SIS User ID'].unique()[:10]:
                student_data = combined_df[combined_df['SIS User ID'] == sid]
                if outcome_filter != "All":
                    fig.add_trace(go.Bar(x=[outcome_filter], y=student_data[outcome_filter], name=f"Student {sid}", marker=dict(line=dict(width=1, color='black'))))
                else:
                    for col in numeric_cols:
                        fig.add_trace(go.Bar(x=[col], y=student_data[col], name=f"Student {sid} - {col}", marker=dict(line=dict(width=1, color='black'))))
            fig.update_layout(
                title=f"{tab_name.upper()} Comparison by Student",
                xaxis_title="Outcomes",
                yaxis_title="Scores",
                barmode='group',
                template="plotly_white",
                height=500,
                margin=dict(t=50, b=50),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
    elif group_by == "Section":
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            sections = combined_df['Source'].str.extract(r'_(\w+)_').iloc[:, 0].unique()
            fig = go.Figure()
            for section in sections:
                section_data = combined_df[combined_df['Source'].str.contains(section)]
                if not section_data.empty:
                    avg_scores = section_data[numeric_cols].mean()
                    if outcome_filter != "All" and outcome_filter in avg_scores.index:
                        fig.add_trace(go.Bar(x=[outcome_filter], y=[avg_scores[outcome_filter]], name=f"Section {section}", marker=dict(line=dict(width=1, color='black'))))
                    else:
                        fig.add_trace(go.Bar(x=avg_scores.index, y=avg_scores.values, name=f"Section {section}", marker=dict(line=dict(width=1, color='black'))))
            fig.update_layout(
                title=f"{tab_name.upper()} Comparison by Section",
                xaxis_title="Outcomes",
                yaxis_title="Average Scores",
                barmode='group',
                template="plotly_white",
                height=500,
                margin=dict(t=50, b=50),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
    elif group_by == "Course":
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        if not numeric_cols.empty:
            courses = combined_df['Source'].str.split('_').str[0].unique()
            fig = go.Figure()
            for course in courses:
                course_data = combined_df[combined_df['Source'].str.startswith(course)]
                if not course_data.empty:
                    avg_scores = course_data[numeric_cols].mean()
                    if outcome_filter != "All" and outcome_filter in avg_scores.index:
                        fig.add_trace(go.Bar(x=[outcome_filter], y=[avg_scores[outcome_filter]], name=f"Course {course}", marker=dict(line=dict(width=1, color='black'))))
                    else:
                        fig.add_trace(go.Bar(x=avg_scores.index, y=avg_scores.values, name=f"Course {course}", marker=dict(line=dict(width=1, color='black'))))
            fig.update_layout(
                title=f"{tab_name.upper()} Comparison by Course",
                xaxis_title="Outcomes",
                yaxis_title="Average Scores",
                barmode='group',
                template="plotly_white",
                height=500,
                margin=dict(t=50, b=50),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)

def streamlit_app():
    st.markdown("""
        <style>
        .main { background-color: #f5f5f5; padding: 20px; border-radius: 10px; }
        .stButton>button { 
            background-color: #4CAF50; 
            color: white; 
            border-radius: 5px; 
            padding: 10px 20px; 
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .stButton>button:hover { background-color: #45a049; }
        .stFileUploader { 
            background-color: #ffffff; 
            padding: 15px; 
            border-radius: 5px; 
            border: 1px solid #ddd; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stSelectbox, .stSlider { 
            background-color: #ffffff; 
            padding: 8px; 
            border-radius: 5px; 
            border: 1px solid #ddd;
        }
        .stTabs { 
            background-color: #ffffff; 
            padding: 15px; 
            border-radius: 5px; 
            border: 1px solid #ddd;
        }
        .stAlert { 
            border-radius: 5px; 
            padding: 10px; 
            margin-bottom: 10px;
        }
        .sidebar .sidebar-content { 
            background-color: #fafafa; 
            padding: 15px; 
            border-right: 1px solid #ddd;
        }
        h1, h2, h3 { 
            color: #2c3e50; 
            font-family: 'Arial', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Assessment Dashboard")
        st.markdown("Upload configuration and input files to analyze outcomes.")
        config_file = st.file_uploader("Upload JSON Config (acat_config.json)", type=["json"], key="config_uploader")
        uploaded_files = st.file_uploader("Upload Excel Files (outcomes, assignments, grades)", type=["xlsx"], accept_multiple_files=True, key="excel_uploader")
        st.markdown("---")
        st.subheader("Processing Log")
        log_container = st.container()
        if st.button("Process Files", key="process_button"):
            if not config_file:
                log_container.error("Please upload a JSON config file.")
                return
            if not uploaded_files:
                log_container.error("Please upload at least one Excel file.")
                return
            with st.spinner("Processing files..."):
                config_path = "temp_config.json"
                try:
                    with open(config_path, "wb") as f:
                        f.write(config_file.getvalue())
                except Exception as e:
                    logger.error(f"Error saving temporary config file: {e}")
                    log_container.error(f"Error saving temporary config file: {e}")
                    return
                config = load_config(config_path)
                if not config or 'courses' not in config:
                    log_container.error("Invalid or empty configuration file.")
                    return
                excel_output_folder = config.get('output', {}).get('excel_folder', 'output')
                os.makedirs(excel_output_folder, exist_ok=True)
                for course in config['courses']:
                    course_name = course.get('course_name')
                    semester = course.get('semester')
                    outcomes_file = course.get('outcomes_file')
                    if not course_name or not semester or not outcomes_file:
                        logger.warning(f"Skipping course due to missing info: {course}")
                        log_container.warning(f"Skipping course due to missing info: {course}")
                        continue
                    outcomes = read_outcomes(outcomes_file)
                    if not outcomes:
                        logger.warning(f"No outcomes found for course {course_name}, skipping.")
                        log_container.warning(f"No outcomes found for course {course_name}, skipping.")
                        continue
                    log_container.info(f"Processing course: {course_name}")
                    log_container.write(f"Outcomes: {outcomes}")
                    for section_data in course.get('sections', []):
                        section = section_data.get('section')
                        if not section:
                            logger.warning("Skipping section with missing section name.")
                            log_container.warning("Skipping section with missing section name.")
                            continue
                        log_container.info(f"Section: {section}")
                        assignments_file = section_data.get('assignments_file', '')
                        assignments_mapping = read_assignments(assignments_file, outcomes) if assignments_file else {}
                        log_container.write(f"Assignments Mappings: {assignments_mapping}")
                        final_outcomes = {outcome: assignments_mapping.get(outcome, []) for outcome in outcomes}
                        log_container.write(f"Final Outcomes: {final_outcomes}")
                        required_columns = set(assignment for criteria in final_outcomes.values() for assignment in criteria)
                        grades_file = section_data.get('grades_file')
                        if not grades_file:
                            logger.warning(f"Grades file missing for section {section} of course {course_name}, skipping.")
                            log_container.warning(f"Grades file missing for section {section} of course {course_name}, skipping.")
                            continue
                        student_data = read_grades(grades_file, required_columns=required_columns)
                        if not student_data:
                            logger.warning(f"No student data found for section {section} of course {course_name}, skipping.")
                            log_container.warning(f"No student data found for section {section} of course {course_name}, skipping.")
                            continue
                        log_container.write(f"Student Data: {list(student_data.keys())}")
                        try:
                            acat = ACAT(course_name, semester, section, final_outcomes, student_data)
                            student_outcomes = acat.compute_course_outcomes()
                            acat.summarize_course_outcomes(student_outcomes)
                            excel_output = os.path.join(excel_output_folder, f"{course_name}_{semester}_{section}_outcomes.xlsx")
                            db_output = os.path.join(config.get('output', {}).get('database_folder', 'db'), f"{course_name}_{semester}_{section}_outcomes.db")
                            os.makedirs(os.path.dirname(db_output), exist_ok=True)
                            acat.save_to_excel(student_outcomes, excel_output)
                            acat.save_to_sqlite(db_output, student_outcomes)
                            po_output_file = compute_program_outcomes(config, course_name, semester, section, excel_output, excel_output_folder)
                            if po_output_file:
                                io_output_file = compute_institutional_outcomes(config, course_name, semester, section, po_output_file, excel_output_folder)
                                if io_output_file:
                                    compute_student_assessments(config, course_name, semester, section, excel_output, po_output_file, io_output_file, excel_output_folder)
                            log_container.success(f"Processed {course_name} {semester} {section}")
                        except Exception as e:
                            logger.error(f"Error processing {course_name} section {section}: {e}")
                            log_container.error(f"Error processing {course_name} section {section}: {e}")

    st.title("Program and Institutional Outcomes Assessment System")
    st.markdown("Analyze course, program, and institutional outcomes with interactive visualizations.")
    output_folder = config.get('output', {}).get('excel_folder', 'output') if 'config' in locals() else 'output'
    excel_files = glob.glob(os.path.join(output_folder, "*.xlsx"))
    if excel_files:
        st.header("Data Analysis and Visualization")
        with st.expander("Filters and Grouping", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                courses = list(set(f.split('_')[0] for f in excel_files))
                course_filter = st.selectbox("Select Course", ["All"] + sorted(courses), key="course_filter")
            with col2:
                sections = list(set(f.split('_')[2].replace('_outcomes.xlsx', '').replace('_student_assessment.xlsx', '') for f in excel_files if course_filter == "All" or f.startswith(course_filter)))
                section_filter = st.selectbox("Select Section", ["All"] + sorted(sections), key="section_filter")
            with col3:
                semesters = list(set(f.split('_')[1] for f in excel_files if course_filter == "All" or f.startswith(course_filter)))
                semester_filter = st.selectbox("Select Semester", ["All"] + sorted(semesters), key="semester_filter")
            col4, col5 = st.columns(2)
            with col4:
                outcome_types = ["All"] + sorted([col for file in excel_files for col in safe_read_excel(file).columns if col.startswith(('CO', 'PO', 'IO'))])
                outcome_filter = st.selectbox("Select Outcome", outcome_types, key="outcome_filter")
            with col5:
                score_range = st.slider("Score Range", min_value=0.0, max_value=100.0, value=(0.0, 100.0), step=1.0, key="score_range")
            group_by = st.selectbox("Group By", ["Student", "Section", "Course"], key="group_by")
        
        tabs = st.tabs(["Course Outcomes", "Program Outcomes", "Institutional Outcomes", "Student Assessments", "Comparisons"])
        for i, tab_name in enumerate(["co", "po", "io", "student_assessment"]):
            with tabs[i]:
                filtered_files = [f for f in excel_files if tab_name in f.lower() and (course_filter == "All" or f.startswith(course_filter)) and (section_filter == "All" or section_filter in f) and (semester_filter == "All" or semester_filter in f)]
                if filtered_files:
                    for file in filtered_files:
                        df = safe_read_excel(file)
                        if df is None:
                            continue
                        if score_range != (0.0, 100.0):
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            for col in numeric_cols:
                                df = df[(df[col] >= score_range[0]) & (df[col] <= score_range[1])]
                        if outcome_filter != "All" and outcome_filter in df.columns:
                            df = df[['SIS User ID', outcome_filter]]
                        st.subheader(f"Data: {os.path.basename(file)}")
                        st.dataframe(df, use_container_width=True)
                        if tab_name != "student_assessment":
                            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                            if not numeric_cols.empty:
                                avg_data = df[df['SIS User ID'] == 'Class Average'][numeric_cols].melt()
                                fig = px.bar(avg_data, x="variable", y="value", title=f"{tab_name.upper()} Class Averages",
                                            color="variable", template="plotly_white", text="value")
                                fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
                                fig.update_layout(showlegend=False, height=400, margin=dict(t=50, b=50))
                                st.plotly_chart(fig, use_container_width=True)
                                dist_data = df[numeric_cols].melt()
                                fig_dist = px.histogram(dist_data, x="value", nbins=20, title=f"{tab_name.upper()} Score Distribution",
                                                      color="variable", template="plotly_white")
                                fig_dist.update_layout(showlegend=False, height=400, margin=dict(t=50, b=50))
                                st.plotly_chart(fig_dist, use_container_width=True)
                        buffer = io.BytesIO()
                        try:
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False)
                            st.download_button(
                                label=f"Download {os.path.basename(file)}",
                                data=buffer,
                                file_name=os.path.basename(file),
                                mime="application/vnd.ms-excel",
                                key=f"download_{file}"
                            )
                        except Exception as e:
                            logger.error(f"Error generating download for {file}: {e}")
                            st.error(f"Error generating download for {file}: {e}")
                else:
                    st.info("No data available for this tab.")
        with tabs[4]:
            st.header("Comparison Visualizations")
            comparison_tabs = st.tabs(["CO Comparisons", "PO Comparisons", "IO Comparisons"])
            for j, comp_tab_name in enumerate(["co", "po", "io"]):
                with comparison_tabs[j]:
                    filtered_files = [f for f in excel_files if comp_tab_name in f.lower() and (course_filter == "All" or f.startswith(course_filter)) and (section_filter == "All" or section_filter in f) and (semester_filter == "All" or semester_filter in f)]
                    dfs = {f: safe_read_excel(f) for f in filtered_files if safe_read_excel(f) is not None}
                    generate_comparison_charts(dfs, comp_tab_name, course_filter, section_filter, semester_filter, outcome_filter, score_range, group_by)
    else:
        st.info("No output files found. Please process files first.")

def main():
    generate_mappings()
    streamlit_app()

if __name__ == "__main__":
    main()
