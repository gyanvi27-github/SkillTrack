<?php
/**
 * alert.php — standalone Twilio alert dispatcher
 * Can be called directly: php alert.php
 * Or hit as: GET api.php?action=alert&id=5  (handled by api.php)
 * 
 * Cron (send alerts every morning at 8am):
 *   0 8 * * * php C:/xampp/htdocs/skilltrack/alert.php >> alert_cron.log 2>&1
 */

require_once __DIR__ . '/vendor/autoload.php'; // composer require twilio/sdk

use Twilio\Rest\Client;

$sid      = getenv('TWILIO_SID')        ?: 'ACxxxxxxxx';
$token    = getenv('TWILIO_TOKEN')      ?: 'your_token';
$from     = getenv('TWILIO_FROM')       ?: 'whatsapp:+14155238886';
$to       = getenv('COUNSELOR_PHONE')   ?: 'whatsapp:+91XXXXXXXXXX';

$jsonFile = __DIR__ . '/burnout_predictions.json';
if (!file_exists($jsonFile)) { die("No predictions file found.\n"); }

$data     = json_decode(file_get_contents($jsonFile), true);
$atRisk   = array_filter($data['students'], fn($s) => $s['alert_counselor'] === true);

if (empty($atRisk)) {
    echo date('Y-m-d H:i:s') . " — No at-risk students today.\n";
    exit;
}

$client = new Client($sid, $token);

foreach ($atRisk as $s) {
    $conf = round($s['confidence_score'] * 100, 1);
    $body = "🚨 SkillTrack | {$s['name']} — {$s['risk_level']} RISK\n"
          . "Burnout confidence: {$conf}%\n"
          . "7d mood avg: {$s['avg_mood_7d']}/10 | Trend: {$s['trend']}\n"
          . "Est. days: " . ($s['days_to_burnout_est'] ?? 'N/A') . "\n"
          . "Please schedule a check-in today.";

    try {
        $msg = $client->messages->create($to, ['from' => $from, 'body' => $body]);
        echo date('H:i:s') . " ✓ Alert sent for {$s['name']} — SID: {$msg->sid}\n";
    } catch (Exception $e) {
        echo date('H:i:s') . " ✗ Failed for {$s['name']}: {$e->getMessage()}\n";
    }
}
