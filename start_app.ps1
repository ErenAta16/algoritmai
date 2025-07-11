# AI Algoritma Danışmanı - PowerShell Başlatıcı
# Bu script backend ve frontend'i sırayla başlatır

param(
    [switch]$Verbose
)

# UTF-8 encoding ayarla
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Write-Status {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    
    $colors = @{
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Red" = "Red"
        "Blue" = "Cyan"
        "White" = "White"
    }
    
    $color = $colors[$Color]
    Write-Host $Message -ForegroundColor $color
}

function Test-Dependency {
    param(
        [string]$Command,
        [string]$Name,
        [string]$InstallUrl = ""
    )
    
    try {
        $null = Get-Command $Command -ErrorAction Stop
        Write-Status "✅ $Name mevcut" "Green"
        return $true
    }
    catch {
        Write-Status "❌ $Name bulunamadı" "Red"
        if ($InstallUrl) {
            Write-Status "Yükleyin: $InstallUrl" "Yellow"
        }
        return $false
    }
}

function Start-Backend {
    Write-Status "🚀 Backend başlatılıyor..." "Blue"
    
    try {
        $backendJob = Start-Job -ScriptBlock {
            Set-Location $using:PWD
            python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
        } -Name "Backend"
        
        # Backend'in başlaması için bekle
        Start-Sleep -Seconds 5
        
        if ($backendJob.State -eq "Running") {
            Write-Status "✅ Backend başarıyla başlatıldı (http://localhost:8000)" "Green"
            return $backendJob
        } else {
            $result = Receive-Job $backendJob
            Write-Status "❌ Backend başlatılamadı: $result" "Red"
            return $null
        }
    }
    catch {
        Write-Status "❌ Backend başlatma hatası: $_" "Red"
        return $null
    }
}

function Start-Frontend {
    Write-Status "🎨 Frontend başlatılıyor..." "Blue"
    
    $frontendPath = Join-Path $PWD "frontend"
    
    if (-not (Test-Path $frontendPath)) {
        Write-Status "❌ Frontend klasörü bulunamadı: $frontendPath" "Red"
        return $null
    }
    
    try {
        $frontendJob = Start-Job -ScriptBlock {
            Set-Location $using:frontendPath
            npm start
        } -Name "Frontend"
        
        # Frontend'in başlaması için bekle
        Start-Sleep -Seconds 8
        
        if ($frontendJob.State -eq "Running") {
            Write-Status "✅ Frontend başarıyla başlatıldı (http://localhost:3000)" "Green"
            return $frontendJob
        } else {
            $result = Receive-Job $frontendJob
            Write-Status "❌ Frontend başlatılamadı: $result" "Red"
            return $null
        }
    }
    catch {
        Write-Status "❌ Frontend başlatma hatası: $_" "Red"
        return $null
    }
}

function Stop-AllJobs {
    param(
        [array]$Jobs
    )
    
    foreach ($job in $Jobs) {
        if ($job -and $job.State -eq "Running") {
            Write-Status "🛑 $($job.Name) kapatılıyor..." "Yellow"
            Stop-Job $job
            Remove-Job $job
        }
    }
}

# Ana script başlangıcı
Write-Status "=" * 50 "Blue"
Write-Status "🤖 AI Algoritma Danışmanı - PowerShell Başlatıcı" "Blue"
Write-Status "=" * 50 "Blue"
Write-Host ""

Write-Status "🔍 Bağımlılıklar kontrol ediliyor..." "Blue"
Write-Host ""

# Bağımlılıkları kontrol et
$dependenciesOk = $true

if (-not (Test-Dependency "python" "Python" "https://www.python.org/downloads/")) {
    $dependenciesOk = $false
}

if (-not (Test-Dependency "node" "Node.js" "https://nodejs.org/")) {
    $dependenciesOk = $false
}

if (-not (Test-Dependency "npm" "npm")) {
    $dependenciesOk = $false
}

if (-not $dependenciesOk) {
    Write-Status "❌ Bağımlılık kontrolü başarısız" "Red"
    Read-Host "Çıkmak için Enter tuşuna basın"
    exit 1
}

Write-Host ""

# Jobs array'i oluştur
$jobs = @()

try {
    # Backend'i başlat
    $backendJob = Start-Backend
    if ($backendJob) {
        $jobs += $backendJob
    } else {
        throw "Backend başlatılamadı"
    }
    
    Write-Host ""
    
    # Frontend'i başlat
    $frontendJob = Start-Frontend
    if ($frontendJob) {
        $jobs += $frontendJob
    } else {
        throw "Frontend başlatılamadı"
    }
    
    Write-Host ""
    Write-Status "=" * 50 "Green"
    Write-Status "🎉 Uygulama başarıyla başlatıldı!" "Green"
    Write-Status "📱 Frontend: http://localhost:3000" "Green"
    Write-Status "🔧 Backend: http://localhost:8000" "Green"
    Write-Status "📚 API Docs: http://localhost:8000/docs" "Green"
    Write-Status "=" * 50 "Green"
    Write-Host ""
    Write-Status "💡 Çıkmak için Ctrl+C tuşlayın" "Yellow"
    Write-Host ""
    
    # Jobs'ları izle
    while ($jobs | Where-Object { $_.State -eq "Running" }) {
        Start-Sleep -Seconds 1
        
        # Verbose modda job durumlarını göster
        if ($Verbose) {
            foreach ($job in $jobs) {
                if ($job.State -ne "Running") {
                    Write-Status "⚠️ $($job.Name) durdu" "Yellow"
                }
            }
        }
    }
    
}
catch {
    Write-Status "❌ Hata: $_" "Red"
}
finally {
    Write-Status "🛑 Uygulama kapatılıyor..." "Yellow"
    Stop-AllJobs $jobs
    Write-Status "✅ Temizlik tamamlandı" "Green"
} 