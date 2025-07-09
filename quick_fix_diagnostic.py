#!/usr/bin/env python3
"""
Quick System Fix & Diagnostic Tool
"""

import sys
import os
import requests
import json
from pathlib import Path

def test_imports():
    """Test critical imports"""
    print("🔍 Testing Critical Imports...")
    
    required_modules = {
        'fastapi': 'FastAPI framework',
        'uvicorn': 'ASGI server',
        'openai': 'OpenAI API client',
        'pandas': 'Data manipulation',
        'requests': 'HTTP client',
        'joblib': 'Model persistence'
    }
    
    missing = []
    working = []
    
    for module, description in required_modules.items():
        try:
            if module == 'openai':
                import openai
                # Test if it's the new version
                if hasattr(openai, 'OpenAI'):
                    working.append(f"✅ {module} (v1.x - Modern)")
                else:
                    working.append(f"⚠️ {module} (v0.x - Legacy)")
            else:
                __import__(module)
                working.append(f"✅ {module}")
        except ImportError:
            missing.append(f"❌ {module} ({description})")
    
    print("\n📦 Package Status:")
    for item in working:
        print(f"  {item}")
    for item in missing:
        print(f"  {item}")
    
    return len(missing) == 0

def test_backend():
    """Test backend functionality"""
    print("\n🚀 Testing Backend...")
    
    try:
        # Health check
        response = requests.get("http://localhost:5000/health", timeout=3)
        if response.status_code == 200:
            print("  ✅ Backend health: OK")
            health_data = response.json()
            print(f"     Service: {health_data.get('service', 'Unknown')}")
            
            # Quick chat test
            chat_response = requests.post(
                "http://localhost:5000/chat",
                json={"message": "test"},
                timeout=15
            )
            
            if chat_response.status_code == 200:
                chat_data = chat_response.json()
                response_text = chat_data.get("response", "")
                print(f"  ✅ Chat endpoint: OK ({len(response_text)} chars)")
                return True
            else:
                print(f"  ❌ Chat endpoint: HTTP {chat_response.status_code}")
                return False
                
        else:
            print(f"  ❌ Backend health: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("  ❌ Backend: Not running")
        return False
    except requests.exceptions.Timeout:
        print("  ⚠️ Backend: Timeout (may be slow)")
        return False
    except Exception as e:
        print(f"  ❌ Backend error: {str(e)}")
        return False

def test_files():
    """Test essential files"""
    print("\n📁 Testing Essential Files...")
    
    critical_files = [
        ("main.py", "Backend entry point"),
        ("services/openai_service.py", "OpenAI service"),
        ("Algoritma_Veri_Seti.xlsx", "Algorithm dataset"),
        ("professional-test.html", "Test interface")
    ]
    
    all_present = True
    for file_path, description in critical_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} - {description}")
            all_present = False
    
    return all_present

def test_openai_setup():
    """Test OpenAI configuration"""
    print("\n🤖 Testing OpenAI Setup...")
    
    # Check for API key
    api_key = None
    
    # Check .env file
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            content = f.read()
            for line in content.split('\n'):
                if line.startswith('OPENAI_API_KEY'):
                    api_key = line.split('=')[1] if '=' in line else None
                    break
    
    # Check environment
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key and len(api_key.strip()) > 10:
        masked_key = f"sk-...{api_key.strip()[-4:]}"
        print(f"  ✅ API Key: {masked_key}")
        
        # Test OpenAI service
        try:
            # Use proper package import
            from services import OpenAIService
            service = OpenAIService()
            print("  ✅ OpenAI Service: Initialized")
            return True
        except Exception as e:
            print(f"  ❌ OpenAI Service: {str(e)}")
            return False
    else:
        print("  ❌ API Key: Not found or invalid")
        return False

def provide_fix_commands():
    """Provide fix commands if issues found"""
    print("\n🔧 Quick Fix Commands:")
    print("  1. Missing packages:")
    print("     pip install fastapi uvicorn openai pandas requests joblib")
    print("  2. Start backend:")
    print("     python main.py")
    print("  3. Test professional page:")
    print("     Open professional-test.html in browser")

def main():
    print("🚀 Quick System Diagnostic & Fix Tool")
    print("="*50)
    
    # Run tests
    imports_ok = test_imports()
    files_ok = test_files()
    openai_ok = test_openai_setup()
    backend_ok = test_backend()
    
    # Overall status
    print("\n" + "="*50)
    print("📊 QUICK DIAGNOSTIC SUMMARY:")
    
    status_items = [
        ("Imports", imports_ok),
        ("Files", files_ok),
        ("OpenAI", openai_ok),
        ("Backend", backend_ok)
    ]
    
    all_ok = True
    for name, status in status_items:
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}: {'OK' if status else 'ISSUE'}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\n🎉 SYSTEM STATUS: HEALTHY!")
        print("✅ Your professional test page should work perfectly!")
        print("🌐 Professional Test Page: professional-test.html")
        print("🌐 Backend API: http://localhost:5000")
    else:
        print("\n⚠️ SYSTEM STATUS: NEEDS ATTENTION")
        provide_fix_commands()
    
    print("="*50)

if __name__ == "__main__":
    main() 