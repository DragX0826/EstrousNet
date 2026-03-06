<#
.SYNOPSIS
Run EstrousNet end-to-end workflow in smoke or full mode.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/run_workflow.ps1 -Mode smoke -SaveOverlay

.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/run_workflow.ps1 -Mode full -SkipReview
#>

param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke",

    [int]$SmokeImages = 3,
    [int]$MinDistance = 10,
    [int]$MaxPeaks = 500,
    [double]$SeedThresholdRel = 0.2,

    [switch]$TextMode,
    [switch]$SaveOverlay,
    [switch]$SkipReview
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $maxImages = 0
    if ($Mode -eq "smoke") {
        $maxImages = $SmokeImages
    }

    Write-Host "Step 1: Generate cell candidates ($Mode)"
    $genArgs = @(
        "scripts/generate_cell_candidates.py",
        "--min_distance", $MinDistance,
        "--max_peaks", $MaxPeaks,
        "--seed_threshold_rel", $SeedThresholdRel
    )
    if ($maxImages -gt 0) {
        $genArgs += @("--max_images", $maxImages)
    }
    if ($SaveOverlay) {
        $genArgs += @("--save_overlay")
    }
    & python @genArgs
    if ($LASTEXITCODE -ne 0) { throw "generate_cell_candidates failed." }

    if ($SkipReview) {
        Write-Host "Step 2: Review skipped (-SkipReview)."
    }
    else {
        Write-Host "Step 2: Review candidates (annotation)"
        $reviewArgs = @(
            "scripts/review_cell_candidates.py",
            "--input_csv", "data/annotations/candidate_cells.csv",
            "--output_csv", "data/annotations/reviewed_cells.csv"
        )
        if ($TextMode) {
            $reviewArgs += @("--text_mode")
        }
        & python @reviewArgs
        if ($LASTEXITCODE -ne 0) { throw "review_cell_candidates failed." }
    }

    Write-Host "Step 3: Export labeled features"
    & python scripts/export_labeled_features.py `
        --reviewed_csv data/annotations/reviewed_cells.csv `
        --output_csv data/annotations/labeled_cell_features.csv
    if ($LASTEXITCODE -ne 0) { throw "export_labeled_features failed." }

    Write-Host "Step 4: Train cell classifier"
    & python scripts/train_cell_classifier.py `
        --labeled_features_csv data/annotations/labeled_cell_features.csv `
        --output_model_path results/models/cell_type_rf.joblib
    if ($LASTEXITCODE -ne 0) { throw "train_cell_classifier failed." }

    Write-Host "Step 5: Run stage pipeline ($Mode)"
    $runArgs = @(
        "run_pipeline.py",
        "--input_dir", "data/raw",
        "--output_dir", "results",
        "--stage_rules", "config/stage_rules.yaml",
        "--cell_model", "results/models/cell_type_rf.joblib",
        "--min_distance", $MinDistance,
        "--max_peaks", $MaxPeaks,
        "--seed_threshold_rel", $SeedThresholdRel
    )
    if ($maxImages -gt 0) {
        $runArgs += @("--max_images", $maxImages)
    }
    & python @runArgs
    if ($LASTEXITCODE -ne 0) { throw "run_pipeline failed." }

    Write-Host "Workflow completed: $Mode"
}
finally {
    Pop-Location
}
