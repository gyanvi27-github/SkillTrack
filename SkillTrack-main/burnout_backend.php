<?php
/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║          BURNOUT PREDICTION LSTM — PHP BACKEND ENGINE                      ║
 * ║          SkillTrack Student Wellness Intelligence System                    ║
 * ║                                                                              ║
 * ║  ARCHITECTURE:                                                               ║
 * ║  ─────────────────────────────────────────────────────────────────────────  ║
 * ║  PHP handles:                                                                ║
 * ║    1. Load raw CSV → parse & insert into MySQL                               ║
 * ║    2. Feature engineering (rolling avgs, slope, flags)                       ║
 * ║    3. Build 7-day sliding sequences → store in DB                            ║
 * ║    4. Call Python microservice (predict.py) for LSTM inference               ║
 * ║    5. Store confidence scores → serve JSON to dashboard                      ║
 * ║    6. Counselor alert logic                                                  ║
 * ║                                                                              ║
 * ║  Python microservice (predict.py) handles:                                   ║
 * ║    • Only the neural net forward pass (model.predict)                        ║
 * ║    • Receives sequence JSON via stdin, returns score via stdout              ║
 * ║                                                                              ║
 * ║  REQUIREMENTS:                                                               ║
 * ║    • PHP 8.0+, MySQL 5.7+, Python 3.8+                                       ║
 * ║    • pip install tensorflow scikit-learn numpy joblib                        ║
 * ║    • LiteSpeed / Apache on Hostinger                                         ║
 * ║                                                                              ║
 * ║  USAGE (CLI):                                                                ║
 * ║    php burnout_backend.php --action=import --csv=student_mood_checkins.csv   ║
 * ║    php burnout_backend.php --action=engineer                                 ║
 * ║    php burnout_backend.php --action=predict                                  ║
 * ║    php burnout_backend.php --action=report                                   ║
 * ║    php burnout_backend.php --action=all --csv=student_mood_checkins.csv      ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

declare(strict_types=1);
error_reporting(E_ALL);
ini_set('display_errors', '1');
set_time_limit(0); // long-running ML pipeline

// ──────────────────────────────────────────────────────────────────────────────
// CONFIGURATION
// ──────────────────────────────────────────────────────────────────────────────
define('DB_HOST',     'localhost');
define('DB_NAME',     'skilltrack_db');
define('DB_USER',     'root');          // change for production
define('DB_PASS',     '');             // change for production
define('DB_CHARSET',  'utf8mb4');

define('SEQ_LEN',              7);
define('BURNOUT_THRESHOLD',    0.50);
define('HIGH_RISK_THRESHOLD',  0.70);

// Path to the Python predict microservice script (same directory)
define('PREDICT_SCRIPT', __DIR__ . DIRECTORY_SEPARATOR . 'predict.py');
define('PYTHON_BIN', 'C:\\Users\\Dell\\AppData\\Local\\Programs\\Python\\Python313\\python.exe');  // or 'python' on Windows

// Base + engineered feature columns (must match Python training order)
define('BASE_FEATURES', [
    'mood_score', 'sleep_hours', 'study_hours', 'stress_level',
    'social_interactions', 'assignment_load', 'skipped_class',
]);
define('ROLLING_FEATURES', [
    'mood_3d_avg', 'stress_3d_avg', 'mood_7d_avg',
    'mood_slope_3d', 'sleep_deficit_flag',
]);
define('ALL_FEATURES', array_merge(BASE_FEATURES, ROLLING_FEATURES));


// ──────────────────────────────────────────────────────────────────────────────
// DATABASE CONNECTION
// ──────────────────────────────────────────────────────────────────────────────
function getDB(): PDO
{
    static $pdo = null;
    if ($pdo !== null) return $pdo;

    $dsn = sprintf(
        'mysql:host=%s;dbname=%s;charset=%s',
        DB_HOST, DB_NAME, DB_CHARSET
    );
    $pdo = new PDO($dsn, DB_USER, DB_PASS, [
        PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_EMULATE_PREPARES   => false,
    ]);
    return $pdo;
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 0 — SETUP MYSQL TABLES
// ──────────────────────────────────────────────────────────────────────────────
function setupTables(): void
{
    $db = getDB();
    log_msg("Setting up MySQL tables …");

    // Raw check-ins from CSV
    $db->exec("
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    ");

    // Engineered features per row
    $db->exec("
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    ");

    // Scaler parameters (min/max per feature for MinMax normalization)
    $db->exec("
        CREATE TABLE IF NOT EXISTS scaler_params (
            feature_name VARCHAR(50) PRIMARY KEY,
            feature_min  FLOAT NOT NULL,
            feature_max  FLOAT NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    ");

    // Prediction results per student
    $db->exec("
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
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    ");

    log_msg("Tables ready.");
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 1 — IMPORT CSV INTO MYSQL
// ──────────────────────────────────────────────────────────────────────────────
function importCSV(string $csvPath): void
{
    if (!file_exists($csvPath)) {
        die("ERROR: CSV not found at: $csvPath\n");
    }

    $db = getDB();
    log_msg("Importing CSV: $csvPath");

    // Truncate existing raw data
    $db->exec("TRUNCATE TABLE student_checkins");
    $db->exec("TRUNCATE TABLE student_features");
    $db->exec("TRUNCATE TABLE scaler_params");
    $db->exec("TRUNCATE TABLE burnout_predictions");

    $handle = fopen($csvPath, 'r');
    $headers = fgetcsv($handle);
    // Strip UTF-8 BOM (common in Excel CSVs on Windows)
    $headers[0] = str_replace("\xEF\xBB\xBF", '', $headers[0]);
    $headers = array_map('trim', $headers);
    log_msg("  CSV headers (" . count($headers) . "): " . implode(', ', $headers));

    $stmt = $db->prepare("
        INSERT INTO student_checkins
            (student_id, name, archetype, date, day_number, mood_score,
             sleep_hours, study_hours, stress_level, social_interactions,
             assignment_load, skipped_class, burnout_risk)
        VALUES
            (:student_id, :name, :archetype, :date, :day_number, :mood_score,
             :sleep_hours, :study_hours, :stress_level, :social_interactions,
             :assignment_load, :skipped_class, :burnout_risk)
    ");

    $count      = 0;
    $skipped    = 0;
    $lineNum    = 1; // header was line 1
    $headerCount = count($headers);
    $db->beginTransaction();

    while (($row = fgetcsv($handle)) !== false) {
        $lineNum++;

        // Skip completely empty lines
        if (count($row) === 1 && trim($row[0]) === '') {
            $skipped++;
            continue;
        }

        // Skip rows with wrong column count and log them
        if (count($row) !== $headerCount) {
            log_msg("  WARNING: Skipping line $lineNum — expected $headerCount columns, got " . count($row));
            $skipped++;
            continue;
        }

        $data = array_combine($headers, $row);

        $stmt->execute([
            ':student_id'          => (int)($data['student_id'] ?? 0),
            ':name'                => trim($data['name'] ?? ''),
            ':archetype'           => trim($data['archetype'] ?? ''),
            ':date'                => !empty($data['date']) ? $data['date'] : null,
            ':day_number'          => (int)($data['day_number'] ?? 0),
            ':mood_score'          => (float)($data['mood_score'] ?? 0),
            ':sleep_hours'         => (float)($data['sleep_hours'] ?? 0),
            ':study_hours'         => (float)($data['study_hours'] ?? 0),
            ':stress_level'        => (float)($data['stress_level'] ?? 0),
            ':social_interactions' => (float)($data['social_interactions'] ?? 0),
            ':assignment_load'     => (float)($data['assignment_load'] ?? 0),
            ':skipped_class'       => (int)($data['skipped_class'] ?? 0),
            ':burnout_risk'        => (int)($data['burnout_risk'] ?? 0),
        ]);
        $count++;

        if ($count % 500 === 0) {
            $db->commit();
            $db->beginTransaction();
            log_msg("  ... $count rows imported so far");
        }
    }
    $db->commit();
    fclose($handle);

    if ($skipped > 0) {
        log_msg("  WARNING: Skipped $skipped malformed/empty rows.");
    }

    log_msg("  Imported $count rows.");
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 2 — FEATURE ENGINEERING (Rolling averages + slope + flags)
// ──────────────────────────────────────────────────────────────────────────────
function engineerFeatures(): void
{
    $db = getDB();
    log_msg("Engineering features …");

    $db->exec("TRUNCATE TABLE student_features");

    // Get all unique student IDs
    $students = $db->query("SELECT DISTINCT student_id FROM student_checkins ORDER BY student_id")
                   ->fetchAll(PDO::FETCH_COLUMN);

    $insertStmt = $db->prepare("
        INSERT INTO student_features
            (student_id, day_number, mood_score, sleep_hours, study_hours,
             stress_level, social_interactions, assignment_load, skipped_class,
             mood_3d_avg, stress_3d_avg, mood_7d_avg, mood_slope_3d, sleep_deficit_flag)
        VALUES
            (:student_id, :day_number, :mood_score, :sleep_hours, :study_hours,
             :stress_level, :social_interactions, :assignment_load, :skipped_class,
             :mood_3d_avg, :stress_3d_avg, :mood_7d_avg, :mood_slope_3d, :sleep_deficit_flag)
    ");

    $db->beginTransaction();
    $totalRows = 0;

    foreach ($students as $sid) {
        // Load all days for this student, ordered chronologically
        $rows = $db->prepare("
            SELECT day_number, mood_score, sleep_hours, study_hours, stress_level,
                   social_interactions, assignment_load, skipped_class
            FROM   student_checkins
            WHERE  student_id = ?
            ORDER  BY day_number ASC
        ");
        $rows->execute([$sid]);
        $days = $rows->fetchAll();

        $n = count($days);
        for ($i = 0; $i < $n; $i++) {
            $row = $days[$i];

            // ── Rolling 3-day mood avg
            $mood3 = rollingMean($days, $i, 3, 'mood_score');
            // ── Rolling 3-day stress avg
            $stress3 = rollingMean($days, $i, 3, 'stress_level');
            // ── Rolling 7-day mood avg
            $mood7 = rollingMean($days, $i, 7, 'mood_score');
            // ── 3-day linear slope of mood
            $slope = moodSlope3d($days, $i);
            // ── Sleep deficit flag
            $sleepFlag = ($row['sleep_hours'] < 5.0) ? 1 : 0;

            $insertStmt->execute([
                ':student_id'          => (int)$sid,
                ':day_number'          => (int)$row['day_number'],
                ':mood_score'          => (float)$row['mood_score'],
                ':sleep_hours'         => (float)$row['sleep_hours'],
                ':study_hours'         => (float)$row['study_hours'],
                ':stress_level'        => (float)$row['stress_level'],
                ':social_interactions' => (float)$row['social_interactions'],
                ':assignment_load'     => (float)$row['assignment_load'],
                ':skipped_class'       => (int)$row['skipped_class'],
                ':mood_3d_avg'         => round($mood3, 4),
                ':stress_3d_avg'       => round($stress3, 4),
                ':mood_7d_avg'         => round($mood7, 4),
                ':mood_slope_3d'       => round($slope, 4),
                ':sleep_deficit_flag'  => $sleepFlag,
            ]);
            $totalRows++;
        }

        if ($totalRows % 500 === 0) {
            $db->commit();
            $db->beginTransaction();
        }
    }
    $db->commit();

    log_msg("  Engineered $totalRows feature rows for " . count($students) . " students.");

    // ── Compute & save MinMax scaler params
    computeAndSaveScalerParams();
}

/**
 * Rolling mean for a given field over the last $window rows up to index $i
 */
function rollingMean(array $rows, int $i, int $window, string $field): float
{
    $start = max(0, $i - $window + 1);
    $slice = array_slice($rows, $start, $i - $start + 1);
    if (empty($slice)) return 0.0;
    return array_sum(array_column($slice, $field)) / count($slice);
}

/**
 * Linear slope of mood over last 3 days (up to index $i)
 * Positive = improving, Negative = declining
 */
function moodSlope3d(array $rows, int $i): float
{
    $start = max(0, $i - 2);
    $window = array_slice($rows, $start, $i - $start + 1);
    $n = count($window);
    if ($n < 2) return 0.0;

    $y = array_column($window, 'mood_score');
    $x = range(0, $n - 1);

    // Least-squares slope: m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    $sumX  = array_sum($x);
    $sumY  = array_sum($y);
    $sumXY = 0.0; $sumX2 = 0.0;
    for ($j = 0; $j < $n; $j++) {
        $sumXY += $x[$j] * $y[$j];
        $sumX2 += $x[$j] * $x[$j];
    }
    $denom = $n * $sumX2 - $sumX * $sumX;
    if (abs($denom) < 1e-10) return 0.0;
    return ($n * $sumXY - $sumX * $sumY) / $denom;
}

/**
 * Compute min/max for every feature across ALL students & save to scaler_params
 */
function computeAndSaveScalerParams(): void
{
    $db = getDB();
    $db->exec("TRUNCATE TABLE scaler_params");

    $features = ALL_FEATURES;
    $stmt = $db->prepare("
        INSERT INTO scaler_params (feature_name, feature_min, feature_max)
        VALUES (:fname, :fmin, :fmax)
    ");

    foreach ($features as $feat) {
        $r = $db->query("SELECT MIN(`$feat`) AS fmin, MAX(`$feat`) AS fmax FROM student_features")
                ->fetch();
        $stmt->execute([
            ':fname' => $feat,
            ':fmin'  => (float)$r['fmin'],
            ':fmax'  => (float)$r['fmax'],
        ]);
    }
    log_msg("  Scaler params saved for " . count($features) . " features.");
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 3 — BUILD SEQUENCES & RUN LSTM PREDICTIONS
// ──────────────────────────────────────────────────────────────────────────────
function runPredictions(): void
{
    $db = getDB();
    log_msg("Computing per-student confidence scores …");

    // Load scaler params
    $scalerRows = $db->query("SELECT feature_name, feature_min, feature_max FROM scaler_params")
                     ->fetchAll();
    $scaler = [];
    foreach ($scalerRows as $r) {
        $scaler[$r['feature_name']] = ['min' => (float)$r['feature_min'], 'max' => (float)$r['feature_max']];
    }

    if (empty($scaler)) {
        die("ERROR: Scaler params not found. Run --action=engineer first.\n");
    }

    $db->exec("TRUNCATE TABLE burnout_predictions");

    $students = $db->query("SELECT DISTINCT student_id FROM student_features ORDER BY student_id")
                   ->fetchAll(PDO::FETCH_COLUMN);

    $insertStmt = $db->prepare("
        INSERT INTO burnout_predictions
            (student_id, name, archetype, confidence_score, risk_level,
             days_to_burnout_est, last_mood, avg_mood_7d, avg_stress_7d,
             avg_sleep_7d, mood_slope_7d, trend, alert_counselor)
        VALUES
            (:student_id, :name, :archetype, :confidence_score, :risk_level,
             :days_to_burnout_est, :last_mood, :avg_mood_7d, :avg_stress_7d,
             :avg_sleep_7d, :mood_slope_7d, :trend, :alert_counselor)
    ");

    // ── Build ALL sequences first, then call Python ONCE (model loads once = fast)
    $allSequences = [];
    foreach ($students as $sid) {
        $rows = $db->prepare("
            SELECT " . implode(', ', array_map(fn($f) => "`$f`", ALL_FEATURES)) . "
            FROM   student_features
            WHERE  student_id = ?
            ORDER  BY day_number DESC
            LIMIT  " . SEQ_LEN
        );
        $rows->execute([$sid]);
        $seqRows = array_reverse($rows->fetchAll());

        if (count($seqRows) < SEQ_LEN) continue;

        $scaledSeq = [];
        foreach ($seqRows as $dayRow) {
            $scaledDay = [];
            foreach (ALL_FEATURES as $feat) {
                $val   = (float)$dayRow[$feat];
                $fmin  = $scaler[$feat]['min'];
                $fmax  = $scaler[$feat]['max'];
                $range = $fmax - $fmin;
                $scaledDay[] = $range > 0 ? ($val - $fmin) / $range : 0.0;
            }
            $scaledSeq[] = $scaledDay;
        }
        $allSequences[(string)$sid] = $scaledSeq;
    }

    // ── Single Python call for ALL students
    log_msg("  Calling Python model for " . count($allSequences) . " students (one call) …");
    $allScores = callPythonPredictBatch($allSequences);
    if (empty($allScores)) {
        die("ERROR: Python prediction returned no scores. Check predict.py\n");
    }

    $db->beginTransaction();
    foreach ($students as $sid) {
        $confidence = $allScores[(string)$sid] ?? null;
        if ($confidence === null) continue;

        // ── Risk level
        if ($confidence >= HIGH_RISK_THRESHOLD) {
            $riskLevel = 'HIGH';
        } elseif ($confidence >= BURNOUT_THRESHOLD) {
            $riskLevel = 'MEDIUM';
        } else {
            $riskLevel = 'LOW';
        }

        // ── Raw stats for last 7 days
        $rawRows = $db->prepare("
            SELECT mood_score, stress_level, sleep_hours, day_number
            FROM   student_checkins
            WHERE  student_id = ?
            ORDER  BY day_number DESC
            LIMIT  7
        ");
        $rawRows->execute([$sid]);
        $rawLast7 = array_reverse($rawRows->fetchAll());

        $moodVals   = array_column($rawLast7, 'mood_score');
        $stressVals = array_column($rawLast7, 'stress_level');
        $sleepVals  = array_column($rawLast7, 'sleep_hours');

        $lastMood    = (float)end($moodVals);
        $avgMood7    = array_sum($moodVals)   / count($moodVals);
        $avgStress7  = array_sum($stressVals) / count($stressVals);
        $avgSleep7   = array_sum($sleepVals)  / count($sleepVals);
        $slope7      = linearSlope(array_values($moodVals));

        // ── Trend label
        if ($slope7 < -0.05) {
            $trend = 'declining';
        } elseif ($slope7 > 0.05) {
            $trend = 'improving';
        } else {
            $trend = 'stable';
        }

        // ── Days-to-burnout estimate
        $daysEst = null;
        if ($slope7 < -0.05 && $lastMood > 3.5) {
            $daysEst = (int)(($lastMood - 3.5) / abs($slope7));
            $daysEst = max(1, min($daysEst, 14));
        } elseif ($confidence >= BURNOUT_THRESHOLD) {
            $daysEst = 5;
        }

        // ── Fetch student meta
        $meta = $db->prepare("SELECT name, archetype FROM student_checkins WHERE student_id = ? LIMIT 1");
        $meta->execute([$sid]);
        $m = $meta->fetch();

        $insertStmt->execute([
            ':student_id'          => (int)$sid,
            ':name'                => $m['name'] ?? '',
            ':archetype'           => $m['archetype'] ?? 'unknown',
            ':confidence_score'    => round($confidence, 4),
            ':risk_level'          => $riskLevel,
            ':days_to_burnout_est' => $daysEst,
            ':last_mood'           => round($lastMood, 2),
            ':avg_mood_7d'         => round($avgMood7, 2),
            ':avg_stress_7d'       => round($avgStress7, 2),
            ':avg_sleep_7d'        => round($avgSleep7, 2),
            ':mood_slope_7d'       => round($slope7, 4),
            ':trend'               => $trend,
            ':alert_counselor'     => (int)($confidence >= BURNOUT_THRESHOLD),
        ]);
    }
    $db->commit();

    $total = $db->query("SELECT COUNT(*) FROM burnout_predictions")->fetchColumn();
    $alerts = $db->query("SELECT COUNT(*) FROM burnout_predictions WHERE alert_counselor = 1")->fetchColumn();
    log_msg("  Predictions stored: $total students | Counselor alerts: $alerts");
}

/**
 * Call predict.py ONCE with ALL student sequences as a batch.
 * Input:  array of [student_id => scaled_sequence]
 * Output: array of [student_id => confidence_score]
 */
function callPythonPredictBatch(array $allSequences): array
{
    $payload = json_encode(['sequences' => $allSequences]);
    // On Windows, escapeshellarg wraps in double-quotes which breaks paths
    // Use raw quoting with double-quotes instead
    $python  = '"' . PYTHON_BIN . '"';
    $script  = '"' . PREDICT_SCRIPT . '"';
    $cmd     = "$python $script";

    $descriptors = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];
    $proc = proc_open($cmd, $descriptors, $pipes);
    if (!is_resource($proc)) {
        log_msg("  ERROR: Could not start Python predict process.");
        return [];
    }

    fwrite($pipes[0], $payload);
    fclose($pipes[0]);

    $output = stream_get_contents($pipes[1]);
    $errors = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    proc_close($proc);

    // Log only real errors, not TF info messages
    if ($errors) {
        $errLines = array_filter(explode("\n", trim($errors)), function($line) {
            $line = trim($line);
            return $line !== '' && !str_contains($line, 'oneDNN') && !str_contains($line, 'I0000')
                && !str_contains($line, 'cpu_feature_guard') && !str_contains($line, 'port.cc')
                && !str_contains($line, 'absl') && !str_contains($line, 'TensorFlow GPU')
                && !str_contains($line, 'ABSL') && !str_contains($line, 'WARNING: All log');
        });
        if (!empty($errLines)) {
            log_msg("  Python error: " . implode(" | ", $errLines));
        }
    }

    $decoded = json_decode(trim($output), true);
    if (!is_array($decoded)) {
        log_msg("  ERROR: Could not parse Python output as JSON: '" . substr($output, 0, 200) . "'");
        return [];
    }
    return $decoded;
}

/**
 * Least-squares slope of a 1-D array
 */
function linearSlope(array $y): float
{
    $n = count($y);
    if ($n < 2) return 0.0;
    $x     = range(0, $n - 1);
    $sumX  = array_sum($x);
    $sumY  = array_sum($y);
    $sumXY = 0.0; $sumX2 = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $sumXY += $x[$i] * $y[$i];
        $sumX2 += $x[$i] * $x[$i];
    }
    $denom = $n * $sumX2 - $sumX * $sumX;
    if (abs($denom) < 1e-10) return 0.0;
    return ($n * $sumXY - $sumX * $sumY) / $denom;
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 4 — GENERATE JSON + CSV REPORT
// ──────────────────────────────────────────────────────────────────────────────
function generateReport(): void
{
    $db = getDB();
    log_msg("Generating reports …");

    $rows = $db->query("
        SELECT * FROM burnout_predictions
        ORDER BY confidence_score DESC
    ")->fetchAll();

    // ── JSON for dashboard
    $auc = getModelAUC(); // loaded from file if available
    $payload = [
        'model_auc'  => $auc,
        'threshold'  => BURNOUT_THRESHOLD,
        'seq_len'    => SEQ_LEN,
        'generated'  => date('c'),
        'students'   => $rows,
    ];
    $jsonPath = __DIR__ . '/burnout_predictions.json';
    $tmpPath  = $jsonPath . '.tmp';
    file_put_contents($tmpPath, json_encode($payload, JSON_PRETTY_PRINT));
    rename($tmpPath, $jsonPath);
    log_msg("  Saved: burnout_predictions.json");

    // ── CSV report
    $csvPath = __DIR__ . '/burnout_report.csv';
    $fp = fopen($csvPath, 'w');
    if (!empty($rows)) {
        fputcsv($fp, array_keys($rows[0]));
        foreach ($rows as $r) fputcsv($fp, $r);
    }
    fclose($fp);
    log_msg("  Saved: burnout_report.csv");

    // ── Console summary table
    echo "\n" . str_repeat('=', 72) . "\n";
    echo "  CONFIDENCE SCORE REPORT\n";
    echo str_repeat('=', 72) . "\n";
    printf("%-4s  %-14s  %6s  %-8s  %5s  %-10s  %5s\n",
        'ID', 'Name', 'Score', 'Risk', 'Days', 'Trend', 'Mood');
    echo str_repeat('-', 72) . "\n";
    foreach ($rows as $r) {
        $flag = $r['risk_level'] === 'HIGH' ? '⚠ ' : ($r['risk_level'] === 'MEDIUM' ? '! ' : '  ');
        $days = $r['days_to_burnout_est'] ?? '—';
        printf("%s%-3d  %-14s  %5.1f%%  %-8s  %5s  %-10s  %5.1f\n",
            $flag,
            $r['student_id'],
            substr($r['name'], 0, 14),
            $r['confidence_score'] * 100,
            $r['risk_level'],
            $days,
            $r['trend'],
            $r['last_mood']
        );
    }

    // Alert summary
    $alertCount = count(array_filter($rows, fn($r) => $r['alert_counselor']));
    echo "\n  Students requiring counselor alert: $alertCount\n";
    foreach ($rows as $r) {
        if ($r['alert_counselor']) {
            sendCounselorAlert($r);
        }
    }
    echo "\n  Pipeline complete.\n" . str_repeat('=', 72) . "\n";
}

/**
 * Read last known AUC from a simple text file written by predict.py training
 */
function getModelAUC(): float
{
    $f = __DIR__ . '/model_auc.txt';
    return file_exists($f) ? (float)trim(file_get_contents($f)) : 0.0;
}


// ──────────────────────────────────────────────────────────────────────────────
// STEP 5 — COUNSELOR ALERT (stub — replace with email/SMS API)
// ──────────────────────────────────────────────────────────────────────────────
function sendCounselorAlert(array $student): void
{
    // ── Stub: replace with mail() or Twilio/SendGrid HTTP call
    log_msg(sprintf(
        "  [Alert stub] %s (ID %d) — %s | %.1f%% confidence | Trend: %s",
        $student['name'],
        $student['student_id'],
        $student['risk_level'],
        $student['confidence_score'] * 100,
        $student['trend']
    ));

    /*
    // ── Email example using PHP mail():
    $to      = 'counselor@yourschool.edu';
    $subject = "SkillTrack Burnout Alert: {$student['name']}";
    $body    = "Student: {$student['name']} (ID {$student['student_id']})\n"
             . "Risk: {$student['risk_level']} ({$student['confidence_score']*100:.1f}%)\n"
             . "Mood trend: {$student['trend']}\n"
             . "Please check in with this student today.";
    mail($to, $subject, $body);
    */
}


// ──────────────────────────────────────────────────────────────────────────────
// HTTP API ENDPOINT (when called via browser / frontend AJAX)
// ──────────────────────────────────────────────────────────────────────────────
function handleHttpRequest(): void
{
    header('Content-Type: application/json');
    header('Access-Control-Allow-Origin: *');

    $action = $_GET['action'] ?? 'predictions';

    switch ($action) {
        case 'predictions':
            // Return latest predictions as JSON
            $file = __DIR__ . '/burnout_predictions.json';
            if (!file_exists($file)) {
                echo json_encode(['error' => 'No predictions yet. Run the pipeline first.']);
                return;
            }
            readfile($file);
            break;

        case 'student':
            $sid  = (int)($_GET['id'] ?? 0);
            $db   = getDB();
            $pred = $db->prepare("SELECT * FROM burnout_predictions WHERE student_id = ?");
            $pred->execute([$sid]);
            $row  = $pred->fetch();

            // Last 30 days of check-ins for the timeline chart
            $hist = $db->prepare("
                SELECT day_number, mood_score, stress_level, sleep_hours, burnout_risk
                FROM   student_checkins
                WHERE  student_id = ?
                ORDER  BY day_number DESC
                LIMIT  30
            ");
            $hist->execute([$sid]);

            echo json_encode([
                'prediction' => $row,
                'history'    => array_reverse($hist->fetchAll()),
            ], JSON_PRETTY_PRINT);
            break;

        case 'summary':
            $db  = getDB();
            $out = $db->query("
                SELECT
                    risk_level,
                    COUNT(*)                     AS count,
                    AVG(confidence_score)*100    AS avg_confidence,
                    AVG(avg_mood_7d)             AS avg_mood
                FROM burnout_predictions
                GROUP BY risk_level
            ")->fetchAll();
            echo json_encode($out, JSON_PRETTY_PRINT);
            break;

        default:
            http_response_code(400);
            echo json_encode(['error' => "Unknown action: $action"]);
    }
}


// ──────────────────────────────────────────────────────────────────────────────
// UTILITY
// ──────────────────────────────────────────────────────────────────────────────
function log_msg(string $msg): void
{
    echo '[' . date('H:i:s') . '] ' . $msg . "\n";
}


// ──────────────────────────────────────────────────────────────────────────────
// ENTRY POINT
// ──────────────────────────────────────────────────────────────────────────────
if (PHP_SAPI === 'cli') {
    // CLI usage: php burnout_backend.php --action=all --csv=data.csv
    $opts    = getopt('', ['action:', 'csv:']);
    $action  = $opts['action'] ?? 'all';
    $csvFile = $opts['csv']    ?? 'student_mood_checkins.csv';

    setupTables();

    switch ($action) {
        case 'import':
            importCSV($csvFile);
            break;
        case 'engineer':
            engineerFeatures();
            break;
        case 'predict':
            runPredictions();
            break;
        case 'report':
            generateReport();
            break;
        case 'all':
        default:
            echo str_repeat('=', 72) . "\n";
            echo "  SKILLTRACK — BURNOUT LSTM BACKEND (PHP)\n";
            echo str_repeat('=', 72) . "\n";
            importCSV($csvFile);
            engineerFeatures();
            runPredictions();
            generateReport();
            break;
    }
} else {
    // HTTP request from browser/frontend
    handleHttpRequest();
}
