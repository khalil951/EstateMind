$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $root ".venv\Scripts\Activate.ps1"
$managePy = Join-Path $root "django_backend\manage.py"

if (-not (Test-Path $venvActivate)) {
    throw "Virtualenv activate script not found at $venvActivate. Activate your environment or create a .venv first."
}

if (-not (Test-Path $managePy)) {
    throw "manage.py not found at $managePy. Are you in the project root?"
}

# Optional: accept host and port arguments: default 127.0.0.1:8001
$bindAddr = $args[0]
if (-not $bindAddr) { $bindAddr = "127.0.0.1:8001" }

(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& $venvActivate)
# Run Django development server
& python $managePy runserver $bindAddr
