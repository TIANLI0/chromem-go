param(
    [string]$BenchPattern = 'BenchmarkCollection_Query_NoContent_(1000|5000|25000|100000)|BenchmarkDotProduct',
    [string[]]$CpuSets = @('1', '8'),
    [int[]]$SimdThresholds = @(0, 1024, 1536),
    [int]$Count = 10,
    [string]$Benchtime = '500ms',
    [string]$OutputDir = 'bench-results'
)

$ErrorActionPreference = 'Stop'

function Write-Step([string]$msg) {
    Write-Host "[bench] $msg" -ForegroundColor Cyan
}

function Invoke-GoBench(
    [string]$Label,
    [string]$Cpu,
    [bool]$EnableSimd,
    [Nullable[int]]$SimdMinLength,
    [string]$OutFile
) {
    Write-Step "Running $Label (cpu=$Cpu, simd=$EnableSimd, threshold=$SimdMinLength)"

    if ($EnableSimd) {
        $env:GOEXPERIMENT = 'simd'
    }
    else {
        Remove-Item Env:GOEXPERIMENT -ErrorAction SilentlyContinue
    }

    if ($null -ne $SimdMinLength) {
        $env:CHROMEM_SIMD_MIN_LENGTH = [string]$SimdMinLength
    }
    else {
        Remove-Item Env:CHROMEM_SIMD_MIN_LENGTH -ErrorAction SilentlyContinue
    }

    $cmd = @(
        'test',
        '-run=^$',
        "-bench=$BenchPattern",
        '-benchmem',
        "-benchtime=$Benchtime",
        "-count=$Count",
        "-cpu=$Cpu"
    )

    & go @cmd *> $OutFile
}

function Invoke-Benchstat([string]$BaseFile, [string]$NewFile, [string]$OutFile) {
    Write-Step "Comparing $(Split-Path -Leaf $BaseFile) vs $(Split-Path -Leaf $NewFile)"

    $benchstatCmd = Get-Command benchstat -ErrorAction SilentlyContinue
    if ($null -ne $benchstatCmd) {
        & benchstat $BaseFile $NewFile *> $OutFile
        return
    }

    & go run golang.org/x/perf/cmd/benchstat@latest $BaseFile $NewFile *> $OutFile
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runDir = Join-Path $OutputDir "run-$timestamp"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

Write-Step "Output directory: $runDir"

foreach ($cpu in $CpuSets) {
    $baseFile = Join-Path $runDir "nosimd-cpu$cpu.txt"
    Invoke-GoBench -Label 'baseline' -Cpu $cpu -EnableSimd:$false -SimdMinLength $null -OutFile $baseFile

    foreach ($threshold in $SimdThresholds) {
        $simdFile = Join-Path $runDir "simd-th$threshold-cpu$cpu.txt"
        Invoke-GoBench -Label "simd-th$threshold" -Cpu $cpu -EnableSimd:$true -SimdMinLength $threshold -OutFile $simdFile

        $compareFile = Join-Path $runDir "compare-th$threshold-cpu$cpu.txt"
        Invoke-Benchstat -BaseFile $baseFile -NewFile $simdFile -OutFile $compareFile
    }
}

Remove-Item Env:GOEXPERIMENT -ErrorAction SilentlyContinue
Remove-Item Env:CHROMEM_SIMD_MIN_LENGTH -ErrorAction SilentlyContinue

Write-Step "Done. Key reports:"
Get-ChildItem -Path $runDir -Filter 'compare-*.txt' | Select-Object -ExpandProperty FullName
