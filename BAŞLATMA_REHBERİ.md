# 🚀 AI Algoritma Danışmanı - Başlatma Rehberi

Bu rehber, AI Algoritma Danışmanı uygulamasını kolayca başlatmanız için oluşturulmuştur.

## 📋 Ön Gereksinimler

Uygulamayı çalıştırmadan önce aşağıdaki yazılımların yüklü olduğundan emin olun:

### 🔧 Python (3.8+)
- [Python İndirme Sayfası](https://www.python.org/downloads/)
- Kurulum sonrası `python --version` komutu ile kontrol edin

### 🟢 Node.js (14+)
- [Node.js İndirme Sayfası](https://nodejs.org/)
- Kurulum sonrası `node --version` komutu ile kontrol edin
- npm otomatik olarak Node.js ile birlikte gelir

### 📦 Python Bağımlılıkları
```bash
pip install -r requirements.txt
```

## 🎯 Başlatma Seçenekleri

### 1. 🐍 Python Script (Önerilen)
```bash
python start_app.py
```

**Özellikler:**
- ✅ Otomatik bağımlılık kontrolü
- ✅ Renkli durum mesajları
- ✅ Hata yönetimi
- ✅ Graceful shutdown
- ✅ Process monitoring

### 2. 🔧 PowerShell Script
```powershell
.\start_app.ps1
```

**Verbose mod için:**
```powershell
.\start_app.ps1 -Verbose
```

**Özellikler:**
- ✅ Windows PowerShell uyumlu
- ✅ Job-based process management
- ✅ UTF-8 encoding desteği
- ✅ Detaylı hata raporlama

### 3. 📜 Batch Dosyası
```cmd
start_app.bat
```

**Özellikler:**
- ✅ Basit ve hızlı
- ✅ Ayrı pencerelerde çalışır
- ✅ Windows CMD uyumlu

## 🌐 Erişim Adresleri

Uygulama başarıyla başlatıldıktan sonra:

| Servis | URL | Açıklama |
|--------|-----|----------|
| 🎨 Frontend | http://localhost:3000 | Ana kullanıcı arayüzü |
| 🔧 Backend | http://localhost:8000 | API sunucusu |
| 📚 API Docs | http://localhost:8000/docs | Swagger dokümantasyonu |
| 🔍 Admin Panel | http://localhost:8000/admin | Yönetici paneli |

## 🔐 Varsayılan Kullanıcı Bilgileri

### 👤 Admin Kullanıcısı
- **Kullanıcı Adı:** admin
- **Şifre:** Admin123!

### 👤 Test Kullanıcısı
- **Kullanıcı Adı:** testuser
- **Şifre:** Test123!

## ⚠️ Sorun Giderme

### Backend Başlatılamıyor
1. Python bağımlılıklarını kontrol edin:
   ```bash
   pip install -r requirements.txt
   ```

2. Port 8000'in kullanımda olup olmadığını kontrol edin:
   ```bash
   netstat -an | findstr :8000
   ```

3. Veritabanı dosyalarının mevcut olduğunu kontrol edin:
   ```bash
   dir *.db
   ```

### Frontend Başlatılamıyor
1. Node.js ve npm'in yüklü olduğunu kontrol edin:
   ```bash
   node --version
   npm --version
   ```

2. Frontend bağımlılıklarını yükleyin:
   ```bash
   cd frontend/ai-algorithm-consultant-frontend
   npm install
   ```

3. Port 3000'in kullanımda olup olmadığını kontrol edin:
   ```bash
   netstat -an | findstr :3000
   ```

### PowerShell Execution Policy Hatası
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 🛑 Uygulamayı Durdurma

### Python Script
- `Ctrl+C` tuşlayın

### PowerShell Script
- `Ctrl+C` tuşlayın

### Batch Dosyası
- Açılan pencereleri manuel olarak kapatın
- Veya `taskkill` komutu ile:
  ```cmd
  taskkill /f /im python.exe
  taskkill /f /im node.exe
  ```

## 📊 Log Dosyaları

Uygulama çalışırken log dosyaları şu konumlarda oluşturulur:

- `logs/` klasöründe backend logları
- Frontend console logları tarayıcıda görüntülenir

## 🔄 Manuel Başlatma

Eğer otomatik scriptler çalışmazsa, manuel olarak başlatabilirsiniz:

### Backend (Terminal 1)
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Terminal 2)
```bash
cd frontend/ai-algorithm-consultant-frontend
npm start
```

## 📞 Destek

Sorun yaşarsanız:
1. Log dosyalarını kontrol edin
2. Bağımlılıkları yeniden yükleyin
3. Port çakışmalarını kontrol edin
4. Firewall ayarlarını kontrol edin

---

**🎉 Başarılı bir şekilde başlatıldıktan sonra http://localhost:3000 adresinden uygulamaya erişebilirsiniz!** 