CREATE DATABASE IF NOT EXISTS student_db;
USE student_db;

CREATE TABLE IF NOT EXISTS students (
    student_id INT PRIMARY KEY,
    name VARCHAR(50),
    attendance_pct DECIMAL(5,2),
    study_hours DECIMAL(4,2),
    internal_marks DECIMAL(5,2),
    assignment_score DECIMAL(5,2),
    result VARCHAR(4)
);