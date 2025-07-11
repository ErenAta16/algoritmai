#!/usr/bin/env python3
"""
AI Algoritma Danışmanı - Otomatik Başlatma Scripti
Bu script backend ve frontend'i sırayla başlatır.
"""

import subprocess
import sys
import os
import time
import threading
import signal
from pathlib import Path

class AppStarter:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
    def print_status(self, message, color="white"):
        """Renkli durum mesajları yazdır"""
        colors = {
            "green": "\033[92m",
            "yellow": "\033[93m", 
            "red": "\033[91m",
            "blue": "\033[94m",
            "white": "\033[97m"
        }
        reset = "\033[0m"
        print(f"{colors.get(color, colors['white'])}{message}{reset}")
    
    def check_dependencies(self):
        """Gerekli bağımlılıkları kontrol et"""
        self.print_status("🔍 Bağımlılıklar kontrol ediliyor...", "blue")
        
        # Python bağımlılıkları kontrol et
        try:
            import fastapi
            import uvicorn
            self.print_status("✅ Backend bağımlılıkları mevcut", "green")
        except ImportError as e:
            self.print_status(f"❌ Backend bağımlılıkları eksik: {e}", "red")
            self.print_status("pip install -r requirements.txt komutunu çalıştırın", "yellow")
            return False
        
        # Node.js ve npm kontrol et
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                self.print_status(f"✅ Node.js mevcut: {result.stdout.strip()}", "green")
            else:
                self.print_status("❌ Node.js bulunamadı", "red")
                return False
        except FileNotFoundError:
            self.print_status("❌ Node.js yüklü değil", "red")
            return False
        
        # npm kontrol et - farklı yolları dene
        npm_paths = [
            "npm",  # PATH'te varsa
            r"C:\Program Files\nodejs\npm.cmd",  # Windows varsayılan yolu
            r"C:\Program Files (x86)\nodejs\npm.cmd"  # 32-bit sistemler için
        ]
        
        npm_found = False
        npm_path = None
        
        for path in npm_paths:
            try:
                result = subprocess.run([path, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    self.print_status(f"✅ npm mevcut: {result.stdout.strip()} ({path})", "green")
                    npm_found = True
                    npm_path = path
                    break
            except FileNotFoundError:
                continue
        
        if not npm_found:
            self.print_status("❌ npm bulunamadı", "red")
            self.print_status("Node.js'i yeniden yükleyin: https://nodejs.org/", "yellow")
            return False
        
        # npm path'ini global olarak sakla
        self.npm_path = npm_path
        
        return True
    
    def start_backend(self):
        """Backend'i başlat"""
        self.print_status("🚀 Backend başlatılıyor...", "blue")
        
        try:
            # Backend'i başlat
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Backend'in başlamasını bekle
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                self.print_status("✅ Backend başarıyla başlatıldı (http://localhost:8000)", "green")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                self.print_status(f"❌ Backend başlatılamadı: {stderr}", "red")
                return False
                
        except Exception as e:
            self.print_status(f"❌ Backend başlatma hatası: {e}", "red")
            return False
    
    def start_frontend(self):
        """Frontend'i başlat"""
        self.print_status("🎨 Frontend başlatılıyor...", "blue")
        
        # Frontend klasörüne geç
        frontend_dir = Path("frontend")
        
        if not frontend_dir.exists():
            self.print_status(f"❌ Frontend klasörü bulunamadı: {frontend_dir}", "red")
            return False
        
        try:
            # Frontend klasörüne geç ve npm start çalıştır
            self.frontend_process = subprocess.Popen(
                [self.npm_path, "start"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Frontend'in başlamasını bekle
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                self.print_status("✅ Frontend başarıyla başlatıldı (http://localhost:3000)", "green")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                self.print_status(f"❌ Frontend başlatılamadı: {stderr}", "red")
                return False
                
        except Exception as e:
            self.print_status(f"❌ Frontend başlatma hatası: {e}", "red")
            return False
    
    def monitor_processes(self):
        """Çalışan process'leri izle"""
        while self.running:
            if self.backend_process and self.backend_process.poll() is not None:
                self.print_status("⚠️ Backend durdu", "yellow")
                break
            if self.frontend_process and self.frontend_process.poll() is not None:
                self.print_status("⚠️ Frontend durdu", "yellow")
                break
            time.sleep(1)
    
    def cleanup(self):
        """Process'leri temizle"""
        self.running = False
        
        if self.backend_process:
            self.print_status("🛑 Backend kapatılıyor...", "yellow")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        
        if self.frontend_process:
            self.print_status("🛑 Frontend kapatılıyor...", "yellow")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
    
    def run(self):
        """Ana çalıştırma fonksiyonu"""
        self.print_status("=" * 50, "blue")
        self.print_status("🤖 AI Algoritma Danışmanı - Otomatik Başlatıcı", "blue")
        self.print_status("=" * 50, "blue")
        
        # Bağımlılıkları kontrol et
        if not self.check_dependencies():
            self.print_status("❌ Bağımlılık kontrolü başarısız", "red")
            return
        
        # Signal handler'ları ayarla
        def signal_handler(signum, frame):
            self.print_status("\n🛑 Kapatma sinyali alındı...", "yellow")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Backend'i başlat
            if not self.start_backend():
                return
            
            # Frontend'i başlat
            if not self.start_frontend():
                self.cleanup()
                return
            
            self.print_status("=" * 50, "green")
            self.print_status("🎉 Uygulama başarıyla başlatıldı!", "green")
            self.print_status("📱 Frontend: http://localhost:3000", "green")
            self.print_status("🔧 Backend: http://localhost:8000", "green")
            self.print_status("📚 API Docs: http://localhost:8000/docs", "green")
            self.print_status("=" * 50, "green")
            self.print_status("💡 Çıkmak için Ctrl+C tuşlayın", "yellow")
            
            # Process'leri izle
            self.monitor_processes()
            
        except KeyboardInterrupt:
            self.print_status("\n🛑 Kullanıcı tarafından durduruldu", "yellow")
        except Exception as e:
            self.print_status(f"❌ Beklenmeyen hata: {e}", "red")
        finally:
            self.cleanup()

if __name__ == "__main__":
    starter = AppStarter()
    starter.run() 