param(
    [Parameter(Mandatory)]
    [ValidateSet('backend', 'frontend')]
    [string]$Component
)

$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
if ([IO.Path]::GetPathRoot($project) -ne 'D:\') {
    throw 'Run this launcher from the project on D:.'
}

$temp = Join-Path $project '.local-runtime\tmp'
New-Item -ItemType Directory -Force -Path $temp | Out-Null
$env:TEMP = $temp
$env:TMP = $temp

Push-Location -LiteralPath $project
try {
    if ($Component -eq 'backend') {
        & .\backend\start-local-memory.ps1
        $env:Path = 'D:\POSTGRESQL\bin;' + $env:Path
        uv run --locked --offline --cache-dir .uv-cache --env-file .env python -m backend.local_web
    } else {
        $node = Get-Command node.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        $nodePath = if ($node) { $node.Source } else { 'D:\CodexCaches\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' }
        if (-not (Test-Path -LiteralPath $nodePath)) {
            throw 'Node.js is not on PATH and the existing Node.js runtime on D: was not found.'
        }
        Set-Location -LiteralPath (Join-Path $project 'frontend')
        & $nodePath .\node_modules\next\dist\bin\next dev --hostname 127.0.0.1
    }
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
