#!/usr/bin/env python3
"""
Professional Deployment Script for AI Algorithm Consultant
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
import json
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deployment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.deployment_config = {
            'python_version': '3.9+',
            'required_packages': [
                'fastapi>=0.104.1',
                'uvicorn[standard]>=0.24.0',
                'openai>=1.51.0',
                'pandas>=2.1.3',
                'numpy>=1.24.3',
                'scikit-learn>=1.3.2',
                'python-dotenv>=1.0.0'
            ],
            'ports': {
                'backend': 5000,
                'frontend': 3000
            },
            'environment': 'production'
        }
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 9:
            logger.error("❌ Python 3.9+ required")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor} found")
        
        # Check if pip is available
        try:
            subprocess.run(['pip', '--version'], capture_output=True, check=True)
            logger.info("✅ pip is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ pip not found")
            return False
        
        # Check if Node.js is available (for frontend)
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, check=True, text=True)
            logger.info(f"✅ Node.js {result.stdout.strip()} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("⚠️ Node.js not found - frontend deployment will be skipped")
        
        return True
    
    def setup_environment(self) -> bool:
        """Setup environment variables and configuration"""
        logger.info("🔧 Setting up environment...")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / '.env'
        env_example = self.project_root / '.env.example'
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            logger.info("✅ .env file created from .env.example")
            logger.warning("⚠️ Please configure your .env file with actual values")
        
        # Create logs directory
        logs_dir = self.project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        logger.info("✅ Logs directory created")
        
        # Create data directory if it doesn't exist
        data_dir = self.project_root / 'data'
        data_dir.mkdir(exist_ok=True)
        logger.info("✅ Data directory ensured")
        
        return True
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Upgrade pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Install requirements
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)], 
                             check=True, capture_output=True)
                logger.info("✅ Dependencies installed from requirements.txt")
            else:
                # Install essential packages
                essential_packages = [
                    'fastapi>=0.104.1',
                    'uvicorn[standard]>=0.24.0',
                    'openai>=1.51.0',
                    'pandas>=2.1.3',
                    'numpy>=1.24.3',
                    'scikit-learn>=1.3.2',
                    'python-dotenv>=1.0.0',
                    'pydantic>=2.5.0'
                ]
                
                for package in essential_packages:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                
                logger.info("✅ Essential packages installed")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration files"""
        logger.info("🔍 Validating configuration...")
        
        # Check if algorithm dataset exists
        dataset_paths = [
            'algorithms/Veri_seti.csv',
            'data/Algoritma_Veri_Seti.xlsx',
            'Algoritma_Veri_Seti.xlsx'
        ]
        
        dataset_found = False
        for path in dataset_paths:
            if (self.project_root / path).exists():
                logger.info(f"✅ Algorithm dataset found: {path}")
                dataset_found = True
                break
        
        if not dataset_found:
            logger.error("❌ Algorithm dataset not found")
            return False
        
        # Check if main application files exist
        required_files = [
            'main.py',
            'services/openai_service.py',
            'services/algorithm_recommender.py'
        ]
        
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                logger.error(f"❌ Required file missing: {file_path}")
                return False
        
        logger.info("✅ All required files found")
        return True
    
    def test_application(self) -> bool:
        """Test if application starts correctly"""
        logger.info("🧪 Testing application...")
        
        try:
            # Import main application to test for syntax errors
            sys.path.insert(0, str(self.project_root))
            
            from main import app
            from services.openai_service import OpenAIService
            from services.algorithm_recommender import AlgorithmRecommender
            
            # Test OpenAI service initialization
            openai_service = OpenAIService()
            logger.info("✅ OpenAI service initialized")
            
            # Test Algorithm recommender
            recommender = AlgorithmRecommender()
            logger.info("✅ Algorithm recommender initialized")
            
            # Test basic recommendation
            recommendations = recommender.get_recommendations(
                project_type='classification',
                data_size='medium',
                data_type='numerical'
            )
            
            if recommendations:
                logger.info(f"✅ Basic recommendation test passed: {len(recommendations)} algorithms found")
            else:
                logger.warning("⚠️ No recommendations returned in test")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Application test failed: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create startup script for the application"""
        logger.info("📝 Creating startup script...")
        
        startup_script = self.project_root / 'start.py'
        
        script_content = f'''#!/usr/bin/env python3
"""
Startup script for AI Algorithm Consultant
"""

import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the application"""
    print("🚀 Starting AI Algorithm Consultant...")
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '5000'))
    workers = int(os.getenv('WORKERS', '1'))
    
    # Start server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
'''
        
        with open(startup_script, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(startup_script, 0o755)
        
        logger.info("✅ Startup script created")
        return True
    
    def create_deployment_info(self) -> bool:
        """Create deployment information file"""
        logger.info("📊 Creating deployment info...")
        
        deployment_info = {
            'deployment_date': datetime.now().isoformat(),
            'version': '2.0.0',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'system': os.name,
            'platform': sys.platform,
            'configuration': self.deployment_config,
            'endpoints': {
                'health': '/health',
                'chat': '/chat',
                'recommend': '/recommend',
                'docs': '/docs'
            },
            'features': [
                'GPT-4o Integration',
                'Advanced Algorithm Recommendations',
                'Performance Analytics',
                'Professional UI',
                'Caching System',
                'Rate Limiting',
                'Comprehensive Logging'
            ]
        }
        
        info_file = self.project_root / 'deployment_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(deployment_info, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Deployment info created")
        return True
    
    def deploy(self) -> bool:
        """Main deployment function"""
        logger.info("🚀 Starting deployment process...")
        
        steps = [
            ("System Requirements", self.check_system_requirements),
            ("Environment Setup", self.setup_environment),
            ("Dependencies Installation", self.install_dependencies),
            ("Configuration Validation", self.validate_configuration),
            ("Application Testing", self.test_application),
            ("Startup Script Creation", self.create_startup_script),
            ("Deployment Info Creation", self.create_deployment_info)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"📋 {step_name}...")
            if not step_function():
                logger.error(f"❌ {step_name} failed")
                return False
            logger.info(f"✅ {step_name} completed")
        
        logger.info("🎉 Deployment completed successfully!")
        logger.info("📝 Next steps:")
        logger.info("   1. Configure your .env file with actual values")
        logger.info("   2. Run: python start.py")
        logger.info("   3. Access the application at http://localhost:5000")
        logger.info("   4. View API documentation at http://localhost:5000/docs")
        
        return True

def main():
    """Main entry point"""
    print("🤖 AI Algorithm Consultant - Professional Deployment")
    print("=" * 50)
    
    deployment_manager = DeploymentManager()
    
    if deployment_manager.deploy():
        print("\n🎉 Deployment successful!")
        sys.exit(0)
    else:
        print("\n❌ Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 