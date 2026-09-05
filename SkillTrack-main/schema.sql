-- ============================================================
--  SKILLTRACK — BURNOUT LSTM DATABASE SCHEMA
--  Run this once on your MySQL server before the PHP pipeline
-- ============================================================

CREATE DATABASE IF NOT EXISTS skilltrack_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE skilltrack_db;

-- ── 1. Raw check-ins imported from CSV
CREATE TABLE IF NOT EXISTS student_checkins (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    student_id          INT NOT NULL,
    name                VARCHAR(100),
    archetype           VARCHAR(50),
    date                DATE,
    day_number          INT,
    mood_score          FLOAT,
    sleep_hours         FLOAT,
    study_hours         FLOAT,
    stress_level        FLOAT,
    social_interactions FLOAT,
    assignment_load     FLOAT,
    skipped_class       TINYINT(1),
    burnout_risk        TINYINT(1),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_student_day (student_id, day_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 2. Feature-engineered rows (rolling averages, slope, flags)
CREATE TABLE IF NOT EXISTS student_features (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    student_id          INT NOT NULL,
    day_number          INT NOT NULL,
    mood_score          FLOAT,
    sleep_hours         FLOAT,
    study_hours         FLOAT,
    stress_level        FLOAT,
    social_interactions FLOAT,
    assignment_load     FLOAT,
    skipped_class       TINYINT(1),
    mood_3d_avg         FLOAT,
    stress_3d_avg       FLOAT,
    mood_7d_avg         FLOAT,
    mood_slope_3d       FLOAT,
    sleep_deficit_flag  TINYINT(1),
    INDEX idx_student_day (student_id, day_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 3. MinMax scaler parameters (one row per feature)
CREATE TABLE IF NOT EXISTS scaler_params (
    feature_name VARCHAR(50) PRIMARY KEY,
    feature_min  FLOAT NOT NULL,
    feature_max  FLOAT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ── 4. Final prediction results per student
CREATE TABLE IF NOT EXISTS burnout_predictions (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    student_id           INT NOT NULL,
    name                 VARCHAR(100),
    archetype            VARCHAR(50),
    confidence_score     FLOAT,
    risk_level           VARCHAR(10),
    days_to_burnout_est  INT,
    last_mood            FLOAT,
    avg_mood_7d          FLOAT,
    avg_stress_7d        FLOAT,
    avg_sleep_7d         FLOAT,
    mood_slope_7d        FLOAT,
    trend                VARCHAR(20),
    alert_counselor      TINYINT(1),
    generated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_student (student_id),
    INDEX idx_risk (risk_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
