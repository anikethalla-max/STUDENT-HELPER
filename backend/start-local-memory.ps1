$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if ([IO.Path]::GetPathRoot($project) -ne 'D:\') {
    throw 'Run this script from the project on D: to keep the local memory files there.'
}

$runtime = Join-Path $project '.local-runtime'
$temp = Join-Path $runtime 'tmp'
$data = Join-Path $project '.chroma'
$chroma = Join-Path $runtime 'chroma.exe'
if (-not (Test-Path -LiteralPath $chroma -PathType Leaf)) {
    throw 'Put the official Chroma CLI at .local-runtime\chroma.exe. See backend/POSTGRES.md.'
}
New-Item -ItemType Directory -Path $temp, $data -Force | Out-Null
$ollama = (Get-Command ollama -CommandType Application -ErrorAction Stop).Source

function Start-LocalService($Url, $Exe, $Arguments, $LogName) {
    try {
        $null = Invoke-RestMethod $Url -TimeoutSec 2
        return
    } catch {}
    $process = Start-Process -FilePath $Exe -ArgumentList $Arguments -WorkingDirectory $runtime -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtime "$LogName.log") `
        -RedirectStandardError (Join-Path $runtime "$LogName.err.log")
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            $null = Invoke-RestMethod $Url -TimeoutSec 2
            return
        } catch {
            if ($process.HasExited) { break }
            Start-Sleep -Milliseconds 400
        }
    }
    throw "$LogName did not start. Check its logs under $runtime."
}

# Limit storage settings to the child processes launched by this script.
$settings = @{
    TEMP = $temp
    TMP = $temp
    OLLAMA_MODELS = Join-Path $runtime 'models'
    OLLAMA_HOST = '127.0.0.1:11435'
    OLLAMA_NO_CLOUD = '1'
}
$previous = @{}
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    Start-LocalService 'http://127.0.0.1:8000/api/v2/heartbeat' $chroma ('run --host 127.0.0.1 --port 8000 --path "{0}"' -f $data) 'chroma'
    Start-LocalService 'http://127.0.0.1:11435/api/tags' $ollama 'serve' 'ollama'
    Write-Output 'Local Chroma and Ollama are ready on ports 8000 and 11435.'
} finally {
    foreach ($key in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process')
    }
}
