<?php
header('Content-Type: text/plain');

// Sanitize input to prevent command injection
$script = isset($_GET['script']) ? $_GET['script'] : '';
$script = basename($script); // Only allow the filename, not a path

$allowed_scripts = array(
    'generate_investment_report.py',
    'validate_model.py',
    'train_models.py',
    'train_all_models.py'
);

if (!in_array($script, $allowed_scripts)) {
    echo "Error: Script not allowed";
    exit(1);
}

// Execute the script
$command = "python ../" . escapeshellarg($script) . " 2>&1";
$output = array();
$return_var = 0;
exec($command, $output, $return_var);

echo implode("\n", $output);
echo "\nExit code: " . $return_var;
?>
