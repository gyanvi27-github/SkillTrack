<?php
header("Content-Type: application/json");
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: GET, POST, OPTIONS");
header("Access-Control-Allow-Headers: Content-Type");
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') { http_response_code(200); exit; }

$BASE = __DIR__ . '/';
$JSON_FILE   = $BASE . 'burnout_predictions.json';
$CHECKIN_CSV = $BASE . 'student_mood_checkins.csv';
$LOG_FILE    = $BASE . 'alerts_log.json';

function readJson($path) {
    if (!file_exists($path)) return null;
    $raw = file_get_contents($path);

    // Strip UTF-8 BOM if present
    $raw = preg_replace('/^\xEF\xBB\xBF/', '', $raw);

    // Strip all non-printable control characters except \t \n \r
    $raw = preg_replace('/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/', '', $raw);

    // Fix Python's "None" → JSON null, "True/False" → true/false
    $raw = str_replace(['None', 'True', 'False'], ['null', 'true', 'false'], $raw);

    // Convert encoding
    $raw = mb_convert_encoding($raw, 'UTF-8', mb_detect_encoding($raw));

    $decoded = json_decode($raw, true);
    return $decoded;
}

$action = isset($_GET['action']) ? $_GET['action'] : 'predictions';

if ($action === 'debug') {
    global $BASE, $JSON_FILE, $CHECKIN_CSV;
    $raw = file_exists($JSON_FILE) ? file_get_contents($JSON_FILE) : '';
    $decoded = json_decode($raw, true);
    echo json_encode([
        'json_exists'   => file_exists($JSON_FILE),
        'json_size'     => strlen($raw),
        'json_error'    => json_last_error_msg(),
        'student_count' => $decoded ? count($decoded['students'] ?? []) : 0,
        'first_100'     => substr($raw, 0, 100),
        'php_version'   => PHP_VERSION,
        'base'          => $BASE
    ]);
    exit;
}

if ($action === 'stats') {
    global $JSON_FILE, $LOG_FILE;
    $data = readJson($JSON_FILE);
    if (!$data) {
        echo json_encode(['error' => 'cannot read json: ' . json_last_error_msg(), 'path' => $JSON_FILE]);
        exit;
    }
    $students = isset($data['students']) ? $data['students'] : [];
    $high = $med = $low = 0;
    $confSum = 0;
    foreach ($students as $s) {
        $r = isset($s['risk_level']) ? $s['risk_level'] : '';
        if ($r === 'HIGH')   $high++;
        elseif ($r === 'MEDIUM') $med++;
        else $low++;
        $confSum += isset($s['confidence_score']) ? $s['confidence_score'] : 0;
    }
    $total = count($students);
    $avg = $total > 0 ? round($confSum / $total * 100, 1) : 0;
    $alertsToday = 0;
    if (file_exists($LOG_FILE)) {
        $logs = json_decode(file_get_contents($LOG_FILE), true);
        if ($logs) {
            $today = date('Y-m-d');
            foreach ($logs as $l) {
                if (isset($l['time']) && strpos($l['time'], $today) === 0) $alertsToday++;
            }
        }
    }
    echo json_encode([
        'total'          => $total,
        'high_risk'      => $high,
        'watch_list'     => $med,
        'safe_zone'      => $low,
        'avg_confidence' => $avg,
        'alerts_today'   => $alertsToday,
        'model_auc'      => isset($data['model_auc']) ? $data['model_auc'] : null,
        'generated'      => isset($data['generated']) ? $data['generated'] : null
    ]);
    exit;
}

if ($action === 'predictions') {
    global $JSON_FILE;
    $data = readJson($JSON_FILE);
    if (!$data) {
        echo json_encode(['error' => 'cannot read json', 'path' => $JSON_FILE]);
        exit;
    }
    $students = isset($data['students']) ? $data['students'] : [];
    $risk   = isset($_GET['risk']) ? $_GET['risk'] : null;
    $search = isset($_GET['q']) ? strtolower($_GET['q']) : '';
    $out = [];
    foreach ($students as $s) {
        if ($risk && $s['risk_level'] !== $risk) continue;
        if ($search && strpos(strtolower($s['name']), $search) === false) continue;
        $out[] = $s;
    }
    echo json_encode([
        'model_auc' => isset($data['model_auc']) ? $data['model_auc'] : null,
        'generated' => isset($data['generated']) ? $data['generated'] : null,
        'threshold' => isset($data['threshold']) ? $data['threshold'] : 0.5,
        'students'  => $out
    ]);
    exit;
}

if ($action === 'student') {
    global $JSON_FILE;
    $id = isset($_GET['id']) ? (int)$_GET['id'] : 0;
    if (!$id) { echo json_encode(['error'=>'id required']); exit; }
    $data = readJson($JSON_FILE);
    if (!$data) { echo json_encode(['error'=>'cannot read json']); exit; }
    foreach ($data['students'] as $s) {
        if ((int)$s['student_id'] === $id) { echo json_encode($s); exit; }
    }
    echo json_encode(['error'=>'student not found']);
    exit;
}

if ($action === 'alert') {
    global $JSON_FILE, $LOG_FILE;
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') { echo json_encode(['error'=>'POST required']); exit; }
    $body = json_decode(file_get_contents('php://input'), true);
    $id   = isset($body['student_id']) ? (int)$body['student_id'] : 0;
    if (!$id) { echo json_encode(['error'=>'student_id required']); exit; }
    $data = readJson($JSON_FILE);
    $student = null;
    foreach ($data['students'] as $s) {
        if ((int)$s['student_id'] === $id) { $student = $s; break; }
    }
    if (!$student) { echo json_encode(['error'=>'student not found']); exit; }
    $log = file_exists($LOG_FILE) ? json_decode(file_get_contents($LOG_FILE), true) : [];
    if (!$log) $log = [];
    $log[] = [
        'student_id' => $id,
        'name'       => $student['name'],
        'channel'    => isset($body['channel']) ? $body['channel'] : 'whatsapp',
        'risk'       => $student['risk_level'],
        'confidence' => round($student['confidence_score'] * 100, 1),
        'time'       => date('Y-m-d H:i:s'),
        'status'     => 'simulated'
    ];
    file_put_contents($LOG_FILE, json_encode($log, JSON_PRETTY_PRINT));
    echo json_encode(['success'=>true,'status'=>'simulated','student'=>$student['name']]);
    exit;
}

if ($action === 'checkin') {
    global $CHECKIN_CSV;
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') { echo json_encode(['error'=>'POST required']); exit; }
    $body = json_decode(file_get_contents('php://input'), true);
    foreach (['student_id','mood_score','stress_level','sleep_hours'] as $f) {
        if (!isset($body[$f])) { echo json_encode(['error'=>"Missing: $f"]); exit; }
    }
    $row = implode(',', [
        $body['student_id'], date('Y-m-d'),
        $body['mood_score'], $body['sleep_hours'],
        isset($body['energy_level']) ? $body['energy_level'] : $body['mood_score'],
        $body['stress_level'],
        isset($body['social_interactions']) ? $body['social_interactions'] : 3,
        isset($body['assignment_load']) ? $body['assignment_load'] : 3,
        isset($body['skipped_class']) ? $body['skipped_class'] : 0, 0
    ]) . "\n";
    file_put_contents($CHECKIN_CSV, $row, FILE_APPEND | LOCK_EX);
    echo json_encode(['success'=>true,'saved'=>date('Y-m-d H:i:s')]);
    exit;
}

if ($action === 'history') {
    global $CHECKIN_CSV;
    $id = isset($_GET['id']) ? (int)$_GET['id'] : 0;
    if (!$id || !file_exists($CHECKIN_CSV)) { echo json_encode(['history'=>[]]); exit; }
    $rows = []; $fh = fopen($CHECKIN_CSV, 'r'); $header = fgetcsv($fh);
    while (($line = fgetcsv($fh)) !== false) {
        if (!isset($line[0])) continue;
        $row = array_combine($header, $line);
        if ((int)$row['student_id'] === $id) {
            $rows[] = ['date'=>$row['date'],'mood_score'=>(float)$row['mood_score'],'stress_level'=>(float)$row['stress_level'],'sleep_hours'=>(float)$row['sleep_hours'],'burnout_risk'=>(int)$row['burnout_risk']];
        }
    }
    fclose($fh);
    echo json_encode(['student_id'=>$id,'history'=>$rows]);
    exit;
}
if ($action === 'findbreak') {
    $raw = file_get_contents($JSON_FILE);
    $raw = preg_replace('/^\xEF\xBB\xBF/', '', $raw);
    
    // Try to find exact position of JSON error
    $len = strlen($raw);
    $lo = 0; $hi = $len;
    while ($lo < $hi - 1) {
        $mid = (int)(($lo + $hi) / 2);
        if (json_decode(substr($raw, 0, $mid), true) === null) {
            $hi = $mid;
        } else {
            $lo = $mid;
        }
    }
    // Show 200 chars around the break point
    $start = max(0, $lo - 100);
    $snippet = substr($raw, $start, 200);
    echo json_encode([
        'break_at'  => $lo,
        'total_len' => $len,
        'snippet'   => $snippet
    ]);
    exit;
}
http_response_code(404);
echo json_encode(['error'=>'unknown action']);
?>