"""
Refactored OpenAI Service - Cleaner, smaller, and more focused
"""

import os
import logging
import time
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from services.conversation_manager import ConversationManager
from services.response_generator import ResponseGenerator
from services.algorithm_recommender import AlgorithmRecommender

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIServiceRefactored:
    """
    Refactored OpenAI Service - Clean architecture with separated concerns
    """
    
    def __init__(self):
        """Initialize the refactored OpenAI service"""
        # Initialize components
        self.algorithm_recommender = AlgorithmRecommender()
        self.conversation_manager = ConversationManager()
        self.response_generator = ResponseGenerator(self.algorithm_recommender)
        
        # OpenAI client setup
        self._setup_openai_client()
        
        # System prompts
        self.system_prompt = """Sen "AlgoMentor" adında deneyimli bir makine öğrenmesi uzmanısın. 
        Kullanıcılarla samimi ve yardımsever şekilde konuşuyorsun. Türkçe yanıt ver."""
        
        logger.info("🤖 Refactored OpenAI Service initialized successfully")
    
    def _setup_openai_client(self):
        """Setup OpenAI client with proper validation"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key != 'your_openai_api_key_here' and api_key.startswith('sk-'):
            try:
                self.openai_client = OpenAI(api_key=api_key)
                # Test connection
                self.openai_client.models.list()
                self.openai_enabled = True
                logger.info(f"✅ OpenAI API initialized (key: sk-...{api_key[-4:]})")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI API issue: {str(e)[:100]}...")
                self.openai_enabled = False
                self.openai_client = None
        else:
            self.openai_enabled = False
            self.openai_client = None
            logger.warning("⚠️ OpenAI API key not found, using fallback system")
        
        # Clear API key from memory
        if 'api_key' in locals():
            del api_key
    
    def get_chat_response(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Main chat response method - simplified and clean
        """
        try:
            # Update conversation memory
            self.conversation_manager.update_conversation_memory(user_message, conversation_history)
            
            # Analyze user profile
            self.conversation_manager.analyze_user_profile(user_message)
            
            # Extract project context
            project_context = self._extract_project_context(user_message)
            
            # Determine response type
            response_type = self._determine_response_type(user_message, project_context)
            
            # Generate response based on type
            response = self._generate_response(user_message, project_context, response_type)
            
            # Store response for diversity tracking
            self.conversation_manager.store_response_for_diversity(user_message, response['response'])
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in get_chat_response: {str(e)}")
            return self._get_emergency_fallback()
    
    def _extract_project_context(self, user_message: str) -> Dict:
        """Extract project context from user message"""
        context = {
            'project_type': None,
            'data_size': None,
            'data_type': None,
            'complexity': None,
            'mentioned_algorithms': []
        }
        
        text_lower = user_message.lower()
        
        # Project type detection
        if any(word in text_lower for word in ['sınıflandırma', 'classification', 'kategorilere ayır']):
            context['project_type'] = 'classification'
        elif any(word in text_lower for word in ['kümeleme', 'clustering', 'gruplama']):
            context['project_type'] = 'clustering'
        elif any(word in text_lower for word in ['regresyon', 'regression', 'tahmin']):
            context['project_type'] = 'regression'
        elif any(word in text_lower for word in ['anomali', 'outlier', 'anormal']):
            context['project_type'] = 'anomaly_detection'
        
        # Data size detection
        if any(word in text_lower for word in ['küçük', 'az', 'small']) or any(num in text_lower for num in ['100', '500', '1000']):
            context['data_size'] = 'small'
        elif any(word in text_lower for word in ['büyük', 'çok', 'large']) or any(num in text_lower for num in ['100000', '1000000']):
            context['data_size'] = 'large'
        else:
            context['data_size'] = 'medium'
        
        # Data type detection
        if any(word in text_lower for word in ['sayısal', 'numerical', 'number']):
            context['data_type'] = 'numerical'
        elif any(word in text_lower for word in ['kategorik', 'categorical']):
            context['data_type'] = 'categorical'
        elif any(word in text_lower for word in ['metin', 'text', 'kelime']):
            context['data_type'] = 'text'
        elif any(word in text_lower for word in ['görüntü', 'image', 'resim']):
            context['data_type'] = 'image'
        else:
            context['data_type'] = 'numerical'  # Default
        
        return context
    
    def _determine_response_type(self, user_message: str, context: Dict) -> str:
        """Determine the appropriate response type"""
        text_lower = user_message.lower()
        
        # Check if user wants recommendations
        if any(word in text_lower for word in ['öner', 'öneri', 'tavsiye', 'algoritma']):
            if self._has_enough_context(context):
                return 'recommendation'
            else:
                return 'consultation'
        
        # Check if user is asking about specific algorithm
        algorithms = ['xgboost', 'random forest', 'svm', 'neural', 'logistic', 'k-means']
        if any(algo in text_lower for algo in algorithms):
            return 'algorithm_explanation'
        
        # Check if user wants code
        if any(word in text_lower for word in ['kod', 'code', 'örnek', 'implement']):
            return 'code_request'
        
        # Default to consultation
        return 'consultation'
    
    def _has_enough_context(self, context: Dict) -> bool:
        """Check if we have enough context for recommendations"""
        required_fields = ['project_type', 'data_size', 'data_type']
        return all(context.get(field) for field in required_fields)
    
    def _generate_response(self, user_message: str, context: Dict, response_type: str) -> Dict:
        """Generate response based on type"""
        if response_type == 'recommendation':
            return self.response_generator.generate_algorithm_recommendations(context)
        elif response_type == 'consultation':
            return self.response_generator.generate_consultation_response(user_message, context)
        elif response_type == 'algorithm_explanation':
            algorithm = self._extract_algorithm_name(user_message)
            return self.response_generator.generate_algorithm_explanation(algorithm, context)
        elif response_type == 'code_request':
            return self._generate_code_response(user_message, context)
        else:
            return self.response_generator.generate_consultation_response(user_message, context)
    
    def _extract_algorithm_name(self, user_message: str) -> str:
        """Extract algorithm name from user message"""
        text_lower = user_message.lower()
        
        algorithms = {
            'xgboost': ['xgboost', 'xgb'],
            'random forest': ['random forest', 'rf'],
            'svm': ['svm', 'support vector'],
            'neural network': ['neural', 'nn', 'deep learning'],
            'logistic regression': ['logistic'],
            'k-means': ['kmeans', 'k-means'],
            'dbscan': ['dbscan'],
            'linear regression': ['linear regression']
        }
        
        for algorithm, variants in algorithms.items():
            if any(variant in text_lower for variant in variants):
                return algorithm
        
        return 'unknown'
    
    def _generate_code_response(self, user_message: str, context: Dict) -> Dict:
        """Generate code example response"""
        algorithm = self._extract_algorithm_name(user_message)
        
        if algorithm == 'random forest':
            code_example = """
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veriyi yükle ve böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluştur
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Modeli eğit
rf.fit(X_train, y_train)

# Tahmin yap
y_pred = rf.predict(X_test)

# Doğruluğu hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")
```
"""
            return {
                "response": f"🐍 **Random Forest** kod örneği:\n{code_example}",
                "suggestions": [
                    "Parametreleri nasıl ayarlarım?",
                    "Feature importance nasıl görürüm?",
                    "Başka algoritma örneği göster"
                ],
                "success": True
            }
        else:
            return {
                "response": "Kod örneği için hangi algoritma kullanmak istediğinizi belirtir misiniz?",
                "suggestions": [
                    "Random Forest kod örneği",
                    "XGBoost kod örneği",
                    "Logistic Regression örneği"
                ],
                "success": True
            }
    
    def _get_emergency_fallback(self) -> Dict:
        """Emergency fallback response"""
        return {
            "response": "Üzgünüm, şu anda teknik bir sorun yaşıyorum. Lütfen sorunuzu tekrar deneyin.",
            "suggestions": [
                "Makine öğrenmesi projesi yapıyorum",
                "Algoritma önerisi istiyorum",
                "Kod örneği göster"
            ],
            "success": False,
            "ai_powered": False
        } 