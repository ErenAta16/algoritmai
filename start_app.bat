@echo off
chcp 65001 >nul
title AI Algoritma Danışmanı - Otomatik Başlatıcı

echo ==================================================
echo 🤖 AI Algoritma Danışmanı - Otomatik Başlatıcı
echo ==================================================
echo.

echo 🔍 Bağımlılıklar kontrol ediliyor...
echo.

REM Python kontrol et
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python bulunamadı
    echo Python'u yükleyin: https://www.python.org/downloads/
    pause
    exit /b 1
) else (
    echo ✅ Python mevcut
)

REM Node.js kontrol et
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js bulunamadı
    echo Node.js'i yükleyin: https://nodejs.org/
    pause
    exit /b 1
) else (
    echo ✅ Node.js mevcut
)

REM npm kontrol et
npm --version >nul 2>&1
if errorlevel 1 (
    echo ❌ npm bulunamadı
    echo npm'i yükleyin
    pause
    exit /b 1
) else (
    echo ✅ npm mevcut
)

echo.
echo 🚀 Backend başlatılıyor...
echo.

REM Backend'i başlat
start "Backend" cmd /k "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

REM Backend'in başlaması için bekle
timeout /t 5 /nobreak >nul

echo ✅ Backend başlatıldı (http://localhost:8000)
echo.

echo 🎨 Frontend başlatılıyor...
echo.

REM Frontend klasörüne geç ve başlat
cd frontend
if errorlevel 1 (
    echo ❌ Frontend klasörü bulunamadı
    pause
    exit /b 1
)

start "Frontend" cmd /k "npm start"

REM Ana klasöre geri dön
cd ..\..

echo ✅ Frontend başlatıldı (http://localhost:3000)
echo.

echo ==================================================
echo 🎉 Uygulama başarıyla başlatıldı!
echo 📱 Frontend: http://localhost:3000
echo 🔧 Backend: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo ==================================================
echo.
echo 💡 Çıkmak için herhangi bir tuşa basın...
pause >nul 