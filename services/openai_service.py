import os
from openai import OpenAI
from typing import Dict, List, Optional
from services.algorithm_recommender import AlgorithmRecommender
from dotenv import load_dotenv
import random
import time
import json
from functools import lru_cache
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        """
        Advanced Hybrid AI-powered algorithm consultant with intelligent fallback - ASYNC OPTIMIZED
        """
        self.algorithm_recommender = AlgorithmRecommender()
        self.request_count = 0
        self.last_request_time = 0
        self.rate_limit_delay = 1.0  # 1 second between requests
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize OpenAI with modern client - SECURITY ENHANCED
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Validate API key format
        if api_key and api_key != 'your_openai_api_key_here':
            # Basic API key format validation
            if not api_key.startswith('sk-') or len(api_key) < 20:
                logger.error("❌ Invalid OpenAI API key format")
                self.openai_enabled = False
                self.openai_client = None
            else:
                try:
                    self.openai_client = OpenAI(api_key=api_key)
                    # Test the connection
                    self.openai_client.models.list()
                    self.openai_enabled = True
                    # Log without exposing the key
                    logger.info(f"✅ OpenAI API successfully initialized (key: sk-...{api_key[-4:]})")
                except Exception as e:
                    logger.warning(f"⚠️ OpenAI API issue (quota/connection): {str(e)[:100]}...")
                    self.openai_enabled = False
                    self.openai_client = None
        else:
            self.openai_enabled = False
            self.openai_client = None
            logger.warning("⚠️ OpenAI API key not found, using advanced fallback system")
        
        # Clear the API key from memory after use
        if 'api_key' in locals():
            del api_key
        
        # Always use our advanced AI system regardless of OpenAI status
        self.use_advanced_ai = True
        logger.info("🤖 Advanced AI Algorithm Consultant initialized with intelligent conversation engine")
        
        # Enhanced conversation-focused system prompts
        self.algorithm_expert_prompt = """Sen deneyimli, samimi ve yardımsever bir makine öğrenmesi uzmanısın. Adın "AlgoMentor" ve kullanıcılarla gerçek bir arkadaş gibi konuşuyorsun.

Kişiliğin:
- Meraklı ve öğretmeyi seven
- Teknik bilgiyi basit örneklerle anlatan
- Sabırlı ve destekleyici
- Gerçek dünya deneyimlerini paylaşan
- Yaratıcı çözümler öneren

Konuşma tarzın:
- Doğal, akıcı paragraflar halinde konuş
- "Şöyle ki", "Mesela", "Aslında" gibi günlük ifadeler kullan
- Kişisel deneyimlerini paylaş ("Benim deneyimime göre...")
- Merak uyandırıcı sorular sor
- Cesaretlendirici ve pozitif ol

Algoritma bilgilerini şu şekilde sun:
- Önce hikayesini anlat (nasıl ortaya çıktı, neden önemli)
- Gerçek dünya örnekleri ver
- Avantaj/dezavantajları dengeli şekilde açıkla
- Uygulama ipuçları ve püf noktaları paylaş
- Hangi durumda kullanılacağını net belirt

Her zaman Türkçe konuş ve dostane, samimi bir ton kullan. Robotik cevaplar verme!"""

        self.consultation_prompt = """Sen "AlgoMentor" adında deneyimli bir makine öğrenmesi danışmanısın. Kullanıcılarla gerçek bir mentör gibi konuşuyorsun.

Görevin:
- Kullanıcının projesini samimi bir şekilde dinle ve anla
- Meraklı sorular sor ama sorgulama gibi yapma
- Hikayeler ve örneklerle açıkla
- Kişisel deneyimlerini paylaş
- Cesaretlendirici ve destekleyici ol
- Yaratıcı çözümler öner

Konuşma yaklaşımın:
- "Vay, bu çok ilginç bir proje!" gibi doğal tepkiler ver
- "Benim de benzer bir projede çalışmıştım..." diye deneyim paylaş
- "Şöyle bir yaklaşım deneyebiliriz..." diye öneriler sun
- Teknik terimleri günlük dille açıkla
- Kısa listeler yerine akıcı paragraflar kullan

Bilgi toplarken:
- Doğal soru akışı oluştur
- Kullanıcının motivasyonunu anla
- Proje hedeflerini keşfet
- Kısıtlamaları öğren
- Deneyim seviyesini kavra

Her zaman samimi, yardımsever ve konuşkan ol!"""

        # Conversation memory for context
        self.conversation_memory = []
        self.user_profile = {
            'experience_level': 'unknown',
            'preferred_style': 'unknown',
            'project_domain': 'unknown',
            'technical_comfort': 'unknown'
        }
        
        # Response diversity tracking
        self.response_cache = {}
        self.response_variations = {}
        self.conversation_turn = 0
        
        # Recommendation memory and context tracking
        self.recommendation_history = []
        self.user_preferences = {}
        self.conversation_context = {
            'last_recommendations': [],
            'user_selections': [],
            'discussed_algorithms': [],
            'user_feedback': []
        }

    async def get_chat_response_async(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Async version of get_chat_response for better performance
        """
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive operations in thread pool
        result = await loop.run_in_executor(
            self.executor,
            self.get_chat_response,
            user_message,
            conversation_history
        )
        
        return result

    def get_chat_response(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Enhanced conversational AI response with natural dialogue flow and response diversity - PERFORMANCE OPTIMIZED
        """
        try:
            # Handle None or empty messages
            if user_message is None:
                user_message = ""
            
            print(f"\n🔍 Processing: '{user_message}'")
            
            # Increment conversation turn for diversity tracking
            self.conversation_turn += 1
            
            # Update conversation memory
            self._update_conversation_memory(user_message, conversation_history)
            
            # Analyze user profile and adapt response style
            self._analyze_user_profile(user_message)
            
            # Extract project context with conversation awareness
            project_context = self._extract_enhanced_project_context(user_message, conversation_history)
            project_context['conversation_turn'] = self.conversation_turn
            project_context['conversation_length'] = len(self.conversation_memory)
            
            # Add conversation context for memory awareness
            project_context['conversation_context'] = self.conversation_context
            project_context['conversation_memory'] = self.conversation_memory
            
            print(f"📊 Enhanced Context: {project_context}")
            print(f"🧠 Memory: {len(self.conversation_memory)} messages, {len(self.conversation_context['discussed_algorithms'])} algorithms discussed")
            
            # Determine response type based on conversation flow
            response_type = self._determine_response_type(user_message, project_context)
            print(f"🎯 Response Type: {response_type}")
            
            # Generate contextual response with diversity check
            response = self._generate_diverse_response(user_message, project_context, response_type)
            
            # Store response for diversity tracking
            self._store_response_for_diversity(user_message, response['response'])
            
            # Store recommendations if present
            if 'recommendations' in response:
                self.conversation_context['last_recommendations'] = response['recommendations']
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in get_chat_response: {str(e)}")
            return self._get_emergency_fallback()

    def _rate_limit_check(self):
        """Rate limiting implementation"""
        current_time = time.time()
        if current_time - self.last_request_time < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - (current_time - self.last_request_time))
        self.last_request_time = time.time()
        self.request_count += 1

    @lru_cache(maxsize=100)
    def _cached_gpt_request(self, prompt_hash: str, content: str) -> str:
        """Cache GPT responses to avoid repeated API calls"""
        try:
            self._rate_limit_check()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.algorithm_expert_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=1000,
                temperature=0.3,
                timeout=30
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"GPT request failed: {e}")
            raise e
    
    def _generate_diverse_response(self, user_message: str, project_context: Dict, response_type: str) -> Dict:
        """Generate diverse responses based on conversation history"""
        # Check if we've seen similar messages before
        similar_responses = self._find_similar_responses(user_message)
        
        # Add diversity instruction to context
        if similar_responses:
            project_context['diversity_instruction'] = f"Bu soruya daha önce benzer cevap verdin. Şimdi farklı bir yaklaşım, farklı örnekler ve farklı ifadeler kullan. Önceki cevaplarını tekrar etme."
            project_context['previous_responses'] = similar_responses[:3]  # Last 3 similar responses
        
        # Generate response based on type
        if response_type == 'algorithm_selection':
            return self._handle_algorithm_selection(user_message, project_context)
        elif response_type == 'recommendation_response':
            return self._handle_recommendation_response(user_message, project_context)
        elif response_type == 'algorithm_question':
                return self._handle_algorithm_question(user_message, project_context)
        elif response_type == 'code_request':
            return self._generate_code_example(user_message, project_context)
        elif response_type == 'comparison_request':
            return self._generate_performance_comparison(project_context)
        elif response_type == 'recommendation_ready':
            return self._generate_enhanced_recommendations(user_message, project_context)
        else:
            return self._generate_natural_consultation(user_message, project_context)
    
    def _find_similar_responses(self, user_message: str) -> List[str]:
        """Find similar previous responses to avoid repetition"""
        similar_responses = []
        user_words = set(user_message.lower().split())
        
        for cached_message, cached_response in self.response_cache.items():
            cached_words = set(cached_message.lower().split())
            
            # Calculate similarity
            if user_words and cached_words:
                intersection = user_words.intersection(cached_words)
                similarity = len(intersection) / len(user_words.union(cached_words))
                
                if similarity > 0.5:  # 50% similarity threshold
                    similar_responses.append(cached_response)
        
        return similar_responses
    
    def _store_response_for_diversity(self, user_message: str, response: str):
        """Store response for future diversity checking"""
        # Keep only last 20 responses to prevent memory bloat
        if len(self.response_cache) >= 20:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[user_message] = response
    
    def _track_algorithm_mentions(self, user_message: str):
        """Track mentioned algorithms in conversation"""
        text_lower = user_message.lower()
        
        # Algorithm names to track - expanded list
        algorithms = {
            'xgboost': ['xgboost', 'xgb'],
            'random forest': ['random forest', 'rf'],
            'svm': ['svm', 'support vector'],
            'neural network': ['neural network', 'nn', 'deep learning', 'mlp'],
            'logistic regression': ['logistic regression', 'logistic'],
            'naive bayes': ['naive bayes', 'nb'],
            'knn': ['knn', 'k-nearest'],
            'decision tree': ['decision tree', 'dt'],
            'linear regression': ['linear regression'],
            'k-means': ['kmeans', 'k-means'],
            'dbscan': ['dbscan'],
            'optics': ['optics'],
            'mean shift': ['mean shift'],
            'ensemble': ['ensemble'],
            'gradient boosting': ['gradient boosting', 'gbm'],
            'ada boost': ['ada boost', 'adaboost'],
            'lightgbm': ['lightgbm', 'lgb', 'ts features'],
            'prophet': ['prophet', 'facebook'],
            'catboost': ['catboost']
        }
        
        for algo_name, keywords in algorithms.items():
            if any(keyword in text_lower for keyword in keywords):
                if algo_name not in self.conversation_context['discussed_algorithms']:
                    self.conversation_context['discussed_algorithms'].append(algo_name)
                
                # Track user selection/preference
                if any(word in text_lower for word in ['seçmek', 'istiyorum', 'tercih', 'daha iyi', 'kullanmak']):
                    self.conversation_context['user_selections'].append({
                        'algorithm': algo_name,
                        'message': user_message,
                        'timestamp': time.time()
                    })
    
    def _track_user_preferences(self, user_message: str):
        """Track user feedback and preferences"""
        text_lower = user_message.lower()
        
        # Positive feedback
        if any(word in text_lower for word in ['iyi', 'güzel', 'mükemmel', 'harika', 'beğendim', 'evet']):
            self.conversation_context['user_feedback'].append({
                'type': 'positive',
                'message': user_message,
                'timestamp': time.time()
            })
        
        # Negative feedback
        elif any(word in text_lower for word in ['hayır', 'kötü', 'beğenmedim', 'istemiyorum', 'farklı']):
            self.conversation_context['user_feedback'].append({
                'type': 'negative',
                'message': user_message,
                'timestamp': time.time()
            })
        
        # Questions about algorithms
        elif any(word in text_lower for word in ['neden', 'avantaj', 'dezavantaj', 'niye', 'hangisi']):
            self.conversation_context['user_feedback'].append({
                'type': 'question',
                'message': user_message,
                'timestamp': time.time()
            })
    
    def _update_conversation_memory(self, user_message: str, conversation_history: Optional[List[Dict]] = None):
        """Update conversation memory for better context awareness - MEMORY LEAK FIXED"""
        MAX_MEMORY_SIZE = 20  # Maximum number of messages to keep
        MAX_MEMORY_AGE = 3600  # Maximum age in seconds (1 hour)
        
        current_time = time.time()
        
        # Clean old messages from memory
        self.conversation_memory = [
            msg for msg in self.conversation_memory 
            if current_time - msg.get('timestamp', 0) < MAX_MEMORY_AGE
        ]
        
        if conversation_history:
            # Keep only last 10 messages from history
            recent_history = conversation_history[-10:]
            self.conversation_memory = recent_history
        
        # Add current message
        self.conversation_memory.append({
            'role': 'user',
            'content': user_message,
            'timestamp': current_time
        })
        
        # Keep only the most recent messages
        if len(self.conversation_memory) > MAX_MEMORY_SIZE:
            self.conversation_memory = self.conversation_memory[-MAX_MEMORY_SIZE:]
        
        # Clean response cache periodically
        if len(self.response_cache) > 100:
            # Keep only the most recent 50 responses
            cache_items = list(self.response_cache.items())
            self.response_cache = dict(cache_items[-50:])
        
        # Track algorithm mentions and user preferences
        self._track_algorithm_mentions(user_message)
        self._track_user_preferences(user_message)
        
        # Clean old conversation context
        self._clean_conversation_context()
    
    def _clean_conversation_context(self):
        """Clean old data from conversation context to prevent memory leaks"""
        current_time = time.time()
        MAX_FEEDBACK_AGE = 1800  # 30 minutes
        
        # Clean old feedback
        self.conversation_context['user_feedback'] = [
            feedback for feedback in self.conversation_context['user_feedback']
            if current_time - feedback.get('timestamp', 0) < MAX_FEEDBACK_AGE
        ]
        
        # Keep only last 10 user selections
        if len(self.conversation_context['user_selections']) > 10:
            self.conversation_context['user_selections'] = self.conversation_context['user_selections'][-10:]
        
        # Keep only last 20 discussed algorithms
        if len(self.conversation_context['discussed_algorithms']) > 20:
            self.conversation_context['discussed_algorithms'] = self.conversation_context['discussed_algorithms'][-20:]

    def _analyze_user_profile(self, user_message: str):
        """Analyze user's communication style and technical level"""
        text_lower = user_message.lower()
        
        # Detect experience level
        if any(word in text_lower for word in ['yeni başlıyorum', 'başlangıç', 'bilmiyorum', 'öğreniyorum']):
            self.user_profile['experience_level'] = 'beginner'
        elif any(word in text_lower for word in ['deneyimli', 'uzman', 'profesyonel', 'çalışıyorum']):
            self.user_profile['experience_level'] = 'advanced'
        elif any(word in text_lower for word in ['orta', 'biraz', 'temel']):
            self.user_profile['experience_level'] = 'intermediate'
        
        # Detect communication style preference
        if any(word in text_lower for word in ['detaylı', 'açıkla', 'nasıl', 'neden']):
            self.user_profile['preferred_style'] = 'detailed'
        elif any(word in text_lower for word in ['hızlı', 'kısa', 'özet', 'direkt']):
            self.user_profile['preferred_style'] = 'concise'

    def _determine_response_type(self, user_message: str, context: Dict) -> str:
        """Determine the most appropriate response type"""
        text_lower = user_message.lower()
        
        # Check if user is responding to previous recommendations
        if self._is_responding_to_recommendations(user_message):
            return 'recommendation_response'
        
        # Algorithm selection/preference - expanded algorithm list
        if any(word in text_lower for word in ['seçmek', 'istiyorum', 'tercih', 'daha iyi olur', 'kullanmak']):
            algorithms_to_check = [
                'xgboost', 'random forest', 'svm', 'neural', 'logistic',
                'k-means', 'kmeans', 'dbscan', 'optics', 'mean shift',
                'naive bayes', 'decision tree', 'knn', 'linear regression',
                'ensemble', 'gradient boosting', 'ada boost'
            ]
            if any(algo in text_lower for algo in algorithms_to_check):
                return 'algorithm_selection'
        
        # Check if user is asking about previously discussed algorithms
        if self.conversation_context['discussed_algorithms']:
            for algo in self.conversation_context['discussed_algorithms']:
                if algo in text_lower:
                    if any(word in text_lower for word in ['neden', 'avantaj', 'dezavantaj', 'nasıl']):
                        return 'algorithm_question'
        
        # Algorithm-specific questions - check if any algorithm mentioned
        algorithms_to_check = [
            'xgboost', 'random forest', 'svm', 'neural', 'logistic',
            'k-means', 'kmeans', 'dbscan', 'optics', 'mean shift',
            'naive bayes', 'decision tree', 'knn', 'linear regression',
            'ensemble', 'gradient boosting', 'ada boost', 'algoritma'
        ]
        
        if any(algo in text_lower for algo in algorithms_to_check):
            if any(word in text_lower for word in ['nasıl çalışır', 'nedir', 'açıkla', 'anlat', 'avantaj', 'dezavantaj', 'bilgi', 'hakkında']):
                return 'algorithm_question'
        
        # Code requests
        if any(word in text_lower for word in ['kod', 'örnek', 'implement', 'uygula', 'python']):
            return 'code_request'
        
        # Alternative/comparison requests
        if any(word in text_lower for word in ['karşılaştır', 'hangisi', 'fark', 'vs', 'compare', 'başka', 'alternatif', 'farklı']):
            if any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye']):
                return 'recommendation_response'
        
        # Comparison requests
        if any(word in text_lower for word in ['karşılaştır', 'hangisi', 'fark', 'vs', 'compare']):
            return 'comparison_request'
        
        # Ready for recommendations
        if self._is_ready_for_recommendations(context):
            return 'recommendation_ready'
        
        # Default to consultation
        return 'consultation'

    def _is_ready_for_recommendations(self, context: Dict) -> bool:
        """Check if we have enough context for recommendations"""
        required_fields = ['project_type', 'data_size', 'data_type']
        return all(context.get(field) for field in required_fields)
    
    def _is_responding_to_recommendations(self, user_message: str) -> bool:
        """Check if user is responding to previous recommendations"""
        text_lower = user_message.lower()
        
        # Special case: Alternative requests should always be handled as recommendation responses
        if any(word in text_lower for word in ['başka', 'alternatif']) and any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye']):
            return True
        
        # Check if there were recent recommendations
        if not self.conversation_context['last_recommendations']:
            return False
        
        # Response indicators
        response_indicators = [
            'hayır', 'evet', 'daha iyi', 'tercih', 'seçmek', 'istiyorum',
            'farklı', 'başka', 'alternatif', 'neden', 'avantaj', 'dezavantaj'
        ]
        
        return any(indicator in text_lower for indicator in response_indicators)
    
    def _handle_algorithm_selection(self, user_message: str, context: Dict) -> Dict:
        """Handle when user selects a specific algorithm"""
        text_lower = user_message.lower()
        
        # Find selected algorithm
        selected_algorithm = None
        conversation_context = context.get('conversation_context', {})
        user_selections = conversation_context.get('user_selections', [])
        
        # Check if this algorithm was mentioned before
        for selection in user_selections:
            if selection['algorithm'] in text_lower:
                selected_algorithm = selection['algorithm']
                break
        
        if not selected_algorithm:
            # Try to extract from current message - expanded algorithm list
            algorithms = [
                'xgboost', 'random forest', 'svm', 'neural network', 'logistic regression',
                'k-means', 'kmeans', 'dbscan', 'optics', 'mean shift',
                'naive bayes', 'decision tree', 'knn', 'linear regression',
                'ensemble', 'gradient boosting', 'ada boost'
            ]
            for algo in algorithms:
                if algo.replace(' ', '').replace('-', '') in text_lower.replace(' ', '').replace('-', ''):
                    selected_algorithm = algo
                    break
        
        if selected_algorithm:
            # Direct algorithm choice explanation
            return self._explain_algorithm_choice(selected_algorithm, user_message, context)
        else:
            # If no specific algorithm found, still try to be helpful
            return {
                "response": "🤖 Hangi algoritma kullanmak istediğinizi tam anlayamadım. Daha spesifik olabilir misiniz?\n\nÖrneğin:\n• K-means\n• Random Forest\n• XGBoost\n• SVM\n\nHangisini tercih ediyorsunuz?",
                "suggestions": [
                    "K-means kullanmak istiyorum",
                    "Random Forest tercih ediyorum", 
                    "XGBoost seçmek istiyorum",
                    "SVM kullanayım"
                ],
                "success": True
            }
    
    def _handle_recommendation_response(self, user_message: str, context: Dict) -> Dict:
        """Handle user's response to recommendations"""
        text_lower = user_message.lower()
        
        # Get last recommendations from context
        conversation_context = context.get('conversation_context', {})
        last_recs = conversation_context.get('last_recommendations', [])
        
        # Special handling for alternative requests - even without previous recommendations
        if any(word in text_lower for word in ['başka', 'alternatif', 'farklı']) and any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye']):
            # User wants alternative recommendations
            return self._provide_alternative_recommendations(user_message, context, last_recs)
        elif 'hayır' in text_lower or 'farklı' in text_lower:
            # User rejected recommendations, provide alternatives
            return self._provide_alternative_recommendations(user_message, context, last_recs)
        elif 'neden' in text_lower or 'avantaj' in text_lower:
            # User wants explanation
            return self._explain_recommendations(user_message, context, last_recs)
        else:
            # General response to recommendations
            return self._respond_to_recommendation_feedback(user_message, context, last_recs)
    
    def _explain_algorithm_choice(self, algorithm: str, user_message: str, context: Dict) -> Dict:
        """Explain why user's algorithm choice is good/bad"""
        text_lower = user_message.lower()
        
        # Algorithm explanations based on context
        explanations = {
            'xgboost': {
                'pros': [
                    "Mükemmel seçim! XGBoost gerçekten güçlü bir algoritma",
                    "Yüksek doğruluk oranları sağlar",
                    "Overfitting'e karşı dayanıklı",
                    "Özellik önemini gösterir",
                    "Hızlı eğitim ve tahmin"
                ],
                'cons': [
                    "Hiperparametre ayarlaması gerekebilir",
                    "Küçük veri setlerinde overkill olabilir",
                    "Bellek kullanımı yüksek olabilir"
                ],
                'when_good': "Orta/büyük veri setlerinde, yüksek doğruluk istediğinizde",
                'when_bad': "Çok küçük veri setlerinde, basit problemlerde"
            },
            'random forest': {
                'pros': [
                    "Harika bir seçim! Random Forest çok güvenilir",
                    "Overfitting riski düşük",
                    "Yorumlanabilir sonuçlar",
                    "Hiperparametre ayarı minimal"
                ],
                'cons': [
                    "Çok büyük veri setlerinde yavaş olabilir",
                    "Bellek kullanımı yüksek"
                ],
                'when_good': "Dengeli doğruluk ve hız istediğinizde",
                'when_bad': "Çok büyük veri setlerinde, hız kritikse"
            },
            'k-means': {
                'pros': [
                    "Mükemmel seçim! K-means clustering için ideal",
                    "Basit ve anlaşılır algoritma",
                    "Hızlı çalışır, büyük verilerle başa çıkar",
                    "Görselleştirme imkanları mükemmel",
                    "Müşteri segmentasyonu için çok uygun"
                ],
                'cons': [
                    "K değerini (küme sayısını) önceden belirlemek gerekir",
                    "Küresel olmayan kümelerde zorlanabilir",
                    "Outlier'lara (aykırı değer) hassas"
                ],
                'when_good': "Müşteri segmentasyonu, pazar analizi, veri gruplama",
                'when_bad': "Çok karmaşık şekilli kümeler, belirsiz küme sayısı"
            },
            'kmeans': {
                'pros': [
                    "Mükemmel seçim! K-means clustering için ideal",
                    "Basit ve anlaşılır algoritma",
                    "Hızlı çalışır, büyük verilerle başa çıkar",
                    "Görselleştirme imkanları mükemmel",
                    "Müşteri segmentasyonu için çok uygun"
                ],
                'cons': [
                    "K değerini (küme sayısını) önceden belirlemek gerekir",
                    "Küresel olmayan kümelerde zorlanabilir",
                    "Outlier'lara (aykırı değer) hassas"
                ],
                'when_good': "Müşteri segmentasyonu, pazar analizi, veri gruplama",
                'when_bad': "Çok karmaşık şekilli kümeler, belirsiz küme sayısı"
            }
        }
        
        algo_info = explanations.get(algorithm, {})
        
        # Determine if choice is good based on context
        project_type = context.get('project_type', '')
        data_size = context.get('data_size', '')
        
        is_good_choice = True  # Default positive
        
        if algorithm == 'xgboost' and data_size == 'small':
            is_good_choice = False
        
        # Generate response
        if is_good_choice:
            response = f"🎯 **Harika seçim!** {algorithm.title()} sizin projeniz için gerçekten uygun!\n\n"
            response += f"**Neden mükemmel bir seçim:**\n"
            for pro in algo_info.get('pros', [])[:3]:
                response += f"✅ {pro}\n"
            
            response += f"\n**Sizin durumunuzda özellikle iyi çünkü:** {algo_info.get('when_good', 'genel olarak güçlü bir algoritma')}\n\n"
            
            if algo_info.get('cons'):
                response += f"**Dikkat edilmesi gerekenler:**\n"
                for con in algo_info.get('cons', [])[:2]:
                    response += f"⚠️ {con}\n"
        else:
            response = f"🤔 **{algorithm.title()} seçimi hakkında düşünelim...**\n\n"
            response += f"Bu algoritma güçlü ama sizin durumunuzda belki daha basit bir seçenek daha uygun olabilir.\n\n"
            response += f"**Neden farklı düşünüyorum:**\n"
            for con in algo_info.get('cons', [])[:2]:
                response += f"⚠️ {con}\n"
            
            response += f"\n**Alternatif önerim:** Random Forest veya Logistic Regression daha uygun olabilir."
        
        suggestions = [
            f"{algorithm} nasıl implement edilir?",
            "Alternatif algoritma öner",
            "Performans karşılaştırması yap",
            "Kod örneği göster"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
                         "algorithm_discussed": algorithm
         }
    
    def _provide_alternative_recommendations(self, user_message: str, context: Dict, last_recs: List) -> Dict:
        """Provide alternative recommendations when user rejects previous ones"""
        response = "🔄 **Tamam, farklı seçenekler önereyim!**\n\n"
        
        # Add context awareness
        conversation_context = context.get('conversation_context', {})
        user_selections = conversation_context.get('user_selections', [])
        
        if user_selections:
            last_selection = user_selections[-1]
            response += f"Sizin **{last_selection['algorithm']}** tercihinizi de göz önünde bulundurarak, "
            response += f"farklı yaklaşımlar önereyim:\n\n"
        
        # Get alternative algorithms
        alternative_algorithms = [
            "Support Vector Machine (SVM)",
            "Naive Bayes", 
            "K-Nearest Neighbors (KNN)",
            "Decision Tree",
            "Linear Regression"
        ]
        
        # Filter out previously recommended ones
        if last_recs:
            recommended_names = [rec.get('algorithm', rec.get('name', '')).lower() for rec in last_recs]
            alternative_algorithms = [algo for algo in alternative_algorithms 
                                    if not any(rec_name in algo.lower() for rec_name in recommended_names)]
        
        response += "**Alternatif algoritma önerilerim:**\n\n"
        for i, algo in enumerate(alternative_algorithms[:3], 1):
            response += f"{i}. **{algo}**\n"
            response += f"   • Farklı bir yaklaşım sunar\n"
            response += f"   • Sizin durumunuz için de uygun olabilir\n\n"
        
        # Reference previous conversation
        if last_recs:
            response += f"**Not:** Daha önce {len(last_recs)} algoritma önermiştim. "
            response += f"Bu sefer tamamen farklı yaklaşımlar deneyebiliriz.\n\n"
        
        response += "Hangi alternatif sizi daha çok ilgilendiriyor?"
        
        return {
            "response": response,
            "suggestions": ["İlk alternatifi seç", "İkinci alternatifi seç", "Üçüncü alternatifi seç"],
            "success": True
        }
    
    def _explain_recommendations(self, user_message: str, context: Dict, last_recs: List) -> Dict:
        """Explain why previous recommendations were made"""
        if not last_recs:
            return self._generate_natural_consultation(user_message, context)
        
        response = "💡 **Önerilerimin nedenlerini açıklayayım:**\n\n"
        
        # Add context awareness
        conversation_context = context.get('conversation_context', {})
        user_selections = conversation_context.get('user_selections', [])
        
        if user_selections:
            last_selection = user_selections[-1]
            response += f"Daha önce **{last_selection['algorithm']}** algoritmasını tercih ettiğinizi belirtmiştiniz. "
            response += f"Şimdi size neden başka algoritmaları önerdiğimi açıklayayım:\n\n"
        
        for i, rec in enumerate(last_recs[:3], 1):
            algo_name = rec.get('algorithm', rec.get('name', 'Algoritma'))
            confidence = rec.get('confidence_score', rec.get('confidence', 0.8))
            
            response += f"**{i}. {algo_name}** (Uygunluk: {confidence:.0%})\n"
            
            # Context-based explanations
            project_type = context.get('project_type', '')
            data_size = context.get('data_size', '')
            
            if 'xgboost' in algo_name.lower():
                response += f"   🎯 **Neden önerdiğim:** Yüksek performans, {project_type} problemlerinde çok başarılı\n"
                response += f"   ⚡ **Avantajları:** Hızlı, doğru, overfitting'e dayanıklı\n"
            elif 'random forest' in algo_name.lower():
                response += f"   🎯 **Neden önerdiğim:** Güvenilir, yorumlanabilir, {data_size} veri için ideal\n"
                response += f"   ⚡ **Avantajları:** Stabil sonuçlar, az hiperparametre\n"
            elif 'svm' in algo_name.lower():
                response += f"   🎯 **Neden önerdiğim:** Matematiksel olarak güçlü, {project_type} için etkili\n"
                response += f"   ⚡ **Avantajları:** Yüksek boyutlu veriler için iyi\n"
            elif 'mlp' in algo_name.lower() or 'algılayıcı' in algo_name.lower():
                response += f"   🎯 **Neden önerdiğim:** Çok katmanlı yapı, {project_type} için güçlü\n"
                response += f"   ⚡ **Avantajları:** Esnek yapı, kompleks kalıpları öğrenir\n"
            elif 'ensemble' in algo_name.lower():
                response += f"   🎯 **Neden önerdiğim:** Birden fazla modeli birleştir, {project_type} için stabil\n"
                response += f"   ⚡ **Avantajları:** Yüksek doğruluk, overfitting'e dayanıklı\n"
            else:
                response += f"   🎯 **Neden önerdiğim:** {project_type} projeniz için optimize edilmiş\n"
                response += f"   ⚡ **Avantajları:** Sizin veri tipinize uygun\n"
            
            response += f"\n"
        
        # Reference user's selection if they made one
        if user_selections:
            last_selection = user_selections[-1]
            response += f"**Sizin tercihiniz olan {last_selection['algorithm']} hakkında:** "
            response += f"Bu da mükemmel bir seçim! Yukarıdaki önerilerimle karşılaştırabilirsiniz.\n\n"
        
        response += "Bu açıklamalar yardımcı oldu mu? Başka bir şey merak ediyorsanız sorabilirsiniz!"
        
        return {
            "response": response,
            "suggestions": ["Kod örneği göster", "Performans karşılaştır", "Farklı algoritma öner"],
            "success": True
        }
    
    def _respond_to_recommendation_feedback(self, user_message: str, context: Dict, last_recs: List) -> Dict:
        """Respond to general feedback about recommendations"""
        text_lower = user_message.lower()
        
        # Detect specific user intents
        if any(word in text_lower for word in ['diğer', 'başka', 'farklı', 'alternatif', 'daha fazla']):
            return self._provide_more_alternatives(user_message, context, last_recs)
        
        elif any(word in text_lower for word in ['hepsi', 'tüm', 'bütün', 'liste', 'göster']):
            return self._show_comprehensive_list(user_message, context, last_recs)
        
        elif any(word in text_lower for word in ['evet', 'tamam', 'iyi', 'güzel']):
            response = "🎉 **Harika! Seçiminizi beğendiğinize sevindim.**\n\n"
            response += "Şimdi implementasyon aşamasına geçelim. Size yardımcı olabileceğim konular:\n\n"
            response += "• **Kod örnekleri** - Algoritmayı nasıl kullanacağınızı gösterebilirim\n"
            response += "• **Hiperparametre ayarları** - En iyi performans için optimizasyon\n"
            response += "• **Veri hazırlama** - Algoritma için veriyi nasıl hazırlayacağınız\n"
            response += "• **Performans değerlendirme** - Sonuçları nasıl analiz edeceğiniz\n\n"
            response += "Hangi konuda yardım istiyorsunuz?"
            
            suggestions = ["Kod örneği göster", "Hiperparametre ayarları", "Veri hazırlama", "Performans değerlendirme"]
        else:
            response = "🤔 **Anlıyorum, daha fazla bilgi istiyorsunuz.**\n\n"
            response += "Size nasıl yardımcı olabilirim?\n\n"
            response += "• Algoritmaları daha detaylı açıklayayım\n"
            response += "• Farklı seçenekler önereyim\n"
            response += "• Performans karşılaştırması yapayım\n"
            response += "• Spesifik sorularınızı yanıtlayayım\n\n"
            response += "Ne yapmamı istersiniz?"
            
            suggestions = ["Detaylı açıklama", "Farklı seçenekler", "Performans karşılaştırması", "Spesifik soru sor"]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
    
    def _provide_more_alternatives(self, user_message: str, context: Dict, last_recs: List) -> Dict:
        """Provide more alternative algorithms based on user request"""
        response = "🔄 **Tabii ki! Daha fazla algoritma seçeneği sunayım:**\n\n"
        
        # Get project context
        project_type = context.get('project_type', 'classification')
        data_size = context.get('data_size', 'medium')
        
        # Define comprehensive algorithm lists by category
        classification_algorithms = [
            "XGBoost", "Random Forest", "Support Vector Machine (SVM)", 
            "Naive Bayes", "K-Nearest Neighbors (KNN)", "Decision Tree",
            "Logistic Regression", "Neural Network (MLP)", "AdaBoost",
            "Gradient Boosting", "Extra Trees", "LightGBM", "CatBoost"
        ]
        
        regression_algorithms = [
            "Linear Regression", "Ridge Regression", "Lasso Regression",
            "ElasticNet", "Random Forest Regressor", "XGBoost Regressor",
            "Support Vector Regression", "Neural Network Regressor",
            "Polynomial Regression", "Decision Tree Regressor"
        ]
        
        clustering_algorithms = [
            "K-Means", "Hierarchical Clustering", "DBSCAN", "Gaussian Mixture",
            "Spectral Clustering", "Agglomerative Clustering", "Mean Shift",
            "OPTICS", "Birch", "Mini-Batch K-Means"
        ]
        
        # Select appropriate algorithm list
        if project_type == 'classification':
            algorithms = classification_algorithms
        elif project_type == 'regression':
            algorithms = regression_algorithms
        elif project_type == 'clustering':
            algorithms = clustering_algorithms
        else:
            algorithms = classification_algorithms  # Default
        
        # Filter out already recommended algorithms
        if last_recs:
            recommended_names = [rec.get('algorithm', '').lower() for rec in last_recs]
            algorithms = [algo for algo in algorithms 
                         if not any(rec_name in algo.lower() for rec_name in recommended_names)]
        
        # Show first 5 alternatives
        response += f"**{project_type.title()} için diğer seçenekler:**\n\n"
        for i, algo in enumerate(algorithms[:5], 1):
            response += f"{i}. **{algo}**\n"
            
            # Add context-specific benefits
            if 'xgboost' in algo.lower():
                response += f"   • Yüksek performans, gradient boosting\n"
            elif 'random forest' in algo.lower():
                response += f"   • Güvenilir, overfitting'e dayanıklı\n"
            elif 'svm' in algo.lower():
                response += f"   • Matematiksel olarak güçlü\n"
            elif 'naive bayes' in algo.lower():
                response += f"   • Hızlı, basit, etkili\n"
            elif 'knn' in algo.lower():
                response += f"   • Basit, yorumlanabilir\n"
            else:
                response += f"   • {project_type} problemleri için optimize\n"
            
            response += f"\n"
        
        response += f"**Toplam {len(algorithms)} farklı algoritma seçeneğiniz var!**\n\n"
        response += "Hangi algoritma hakkında daha fazla bilgi almak istersiniz?"
        
        suggestions = algorithms[:3]  # First 3 as suggestions
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
    
    def _show_comprehensive_list(self, user_message: str, context: Dict, last_recs: List) -> Dict:
        """Show comprehensive algorithm list"""
        response = "📋 **Kapsamlı Algoritma Listesi:**\n\n"
        
        project_type = context.get('project_type', 'classification')
        
        if project_type == 'classification':
            response += "**🎯 Sınıflandırma Algoritmaları:**\n\n"
            
            response += "**Ensemble Methods:**\n"
            response += "• Random Forest, XGBoost, LightGBM, CatBoost\n"
            response += "• AdaBoost, Gradient Boosting, Extra Trees\n\n"
            
            response += "**Traditional ML:**\n"
            response += "• Support Vector Machine (SVM)\n"
            response += "• Logistic Regression, Naive Bayes\n"
            response += "• K-Nearest Neighbors (KNN)\n"
            response += "• Decision Tree\n\n"
            
            response += "**Deep Learning:**\n"
            response += "• Neural Network (MLP)\n"
            response += "• Convolutional Neural Network (CNN)\n"
            response += "• Recurrent Neural Network (RNN)\n\n"
            
        elif project_type == 'regression':
            response += "**📈 Regresyon Algoritmaları:**\n\n"
            
            response += "**Linear Models:**\n"
            response += "• Linear Regression, Ridge, Lasso\n"
            response += "• ElasticNet, Polynomial Regression\n\n"
            
            response += "**Tree-based:**\n"
            response += "• Random Forest Regressor\n"
            response += "• XGBoost Regressor, Decision Tree\n\n"
            
            response += "**Advanced:**\n"
            response += "• Support Vector Regression\n"
            response += "• Neural Network Regressor\n\n"
            
        else:
            response += "**🔍 Kümeleme Algoritmaları:**\n\n"
            
            response += "**Centroid-based:**\n"
            response += "• K-Means, Mini-Batch K-Means\n\n"
            
            response += "**Hierarchical:**\n"
            response += "• Agglomerative, Hierarchical\n\n"
            
            response += "**Density-based:**\n"
            response += "• DBSCAN, OPTICS\n\n"
        
        response += "**Hangi kategori sizi daha çok ilgilendiriyor?**"
        
        suggestions = ["Ensemble Methods", "Traditional ML", "Deep Learning", "Performans karşılaştırması"]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
 
    def _generate_natural_consultation(self, user_message: str, context: Dict) -> Dict:
        """Generate natural, conversational consultation response"""
        if self.openai_enabled:
            return self._generate_gpt4_natural_consultation(user_message, context)
        else:
            return self._generate_template_natural_consultation(user_message, context)

    def _generate_gpt4_natural_consultation(self, user_message: str, context: Dict) -> Dict:
        """Generate natural consultation using GPT-4"""
        # Prepare rich context for GPT-4
        conversation_summary = self._summarize_conversation()
        user_profile_summary = self._summarize_user_profile()
        
        context_prompt = f"""
Konuşma Özeti: {conversation_summary}

Kullanıcı Profili: {user_profile_summary}

Mevcut Proje Bilgileri:
- Proje türü: {context.get('project_type', 'Henüz belirlenmedi')}
- Veri boyutu: {context.get('data_size', 'Henüz belirlenmedi')}
- Veri türü: {context.get('data_type', 'Henüz belirlenmedi')}
- Kullanım alanı: {context.get('use_case', 'Henüz belirlenmedi')}
- Kısıtlamalar: {', '.join(context.get('constraints', [])) if context.get('constraints') else 'Yok'}

Kullanıcının Son Mesajı: "{user_message}"

Görevin:
1. Kullanıcının mesajına samimi ve doğal bir şekilde cevap ver
2. Eksik bilgileri öğrenmek için yaratıcı sorular sor
3. Kişisel deneyimlerini paylaş
4. Cesaretlendirici ve destekleyici ol
5. Teknik terimleri günlük dille açıkla
6. 2-3 paragraf halinde akıcı bir konuşma yap

Robotik cevaplar verme, gerçek bir mentor gibi konuş!
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.consultation_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=500,
                temperature=0.9,  # Higher temperature for more creative responses
                presence_penalty=0.3,  # Encourage diverse vocabulary
                frequency_penalty=0.3   # Reduce repetition
            )
            
            gpt_response = response.choices[0].message.content
            
            # Generate contextual suggestions
            suggestions = self._generate_natural_suggestions(context, user_message)
            
            return {
                "response": gpt_response,
                "suggestions": suggestions,
                "success": True,
                "ai_powered": True,
                "conversation_stage": self._get_conversation_stage(context)
            }
                
        except Exception as e:
            logger.error(f"GPT-4 consultation failed: {e}")
            return self._generate_template_natural_consultation(user_message, context)

    def _generate_template_natural_consultation(self, user_message: str, context: Dict) -> Dict:
        """Generate natural consultation using enhanced templates"""
        text_lower = user_message.lower()
        
        # Check for diversity instruction
        diversity_mode = context.get('diversity_instruction') is not None
        conversation_turn = context.get('conversation_turn', 1)
        
        # Greeting responses with enhanced diversity and context awareness
        if any(word in text_lower for word in ['merhaba', 'selam', 'hello', 'hi']):
            # Check conversation history for context
            conversation_context = context.get('conversation_context', {})
            discussed_algorithms = conversation_context.get('discussed_algorithms', [])
            user_selections = conversation_context.get('user_selections', [])
            
            # Check if we have diversity instruction
            if diversity_mode:
                responses = [
                    "Tekrar hoş geldiniz! Bu sefer hangi algoritma macerasına çıkacağız? Her konuşmada farklı hikayeler keşfediyoruz ve bu gerçekten keyifli! \n\nBu kez hangi tür bir proje üzerinde kafa yoruyorsunuz? Belki daha önce hiç düşünmediğiniz bir yaklaşım bulabiliriz.",
                    
                    "Yine buradayız! Makine öğrenmesi dünyasında yeni bir keşif yapmaya hazır mısınız? Her seferinde farklı açılardan bakıyoruz ve bu çok eğlenceli! \n\nBu sefer hangi veri bilimi problemini çözmek istiyorsunuz? Belki bambaşka bir algoritma ailesi keşfederiz!",
                    
                    "Geri döndüğünüz için mutluyum! Bu kez hangi algoritma yolculuğuna çıkacağız? Her konuşma yeni perspektifler getiriyor. \n\nBu sefer hangi tür bir analiz yapmayı planlıyorsunuz? Farklı bir yaklaşım denemek için sabırsızlanıyorum!"
                ]
            else:
                base_responses = [
                    "Merhaba! Ben AlgoMentor, makine öğrenmesi algoritmalarında size yardımcı olmak için buradayım. Gerçekten heyecan verici bir alanda çalışıyorsunuz! \n\nBenim deneyimime göre, doğru algoritma seçimi projenin başarısının %80'ini belirliyor. Peki, hangi tür bir proje üzerinde çalışıyorsunuz? Merak ettim çünkü her projenin kendine özgü güzellikleri var.",
                    
                    "Selam! Hoş geldiniz! Ben makine öğrenmesi dünyasında size rehberlik edecek AlgoMentor'unuz. Yıllardır bu alanda çalışıyorum ve her yeni proje beni hala heyecanlandırıyor.\n\nŞöyle ki, algoritma seçimi biraz müzik enstrümanı seçmeye benziyor - her biri farklı melodiler çıkarıyor. Sizin projeniz hangi tür bir 'melodi' çıkarmak istiyor? Anlatsanız, size en uygun 'enstrümanı' bulalım!"
                ]
                
                # Add context awareness if there's conversation history
                if discussed_algorithms or user_selections:
                    context_addition = "\n\n**Konuşma geçmişimizden:** "
                    if discussed_algorithms:
                        context_addition += f"Daha önce {', '.join(discussed_algorithms)} algoritmalarını konuşmuştuk. "
                    if user_selections:
                        last_selection = user_selections[-1]
                        context_addition += f"Özellikle {last_selection['algorithm']} algoritmasını tercih ettiğinizi hatırlıyorum. "
                    context_addition += "Bu bilgileri göz önünde bulundurarak size yardımcı olabilirim!"
                    
                    responses = [resp + context_addition for resp in base_responses]
                else:
                    responses = base_responses
            
            response = random.choice(responses)
            suggestions = [
                "Veri sınıflandırması yapmak istiyorum",
                "Tahmin modeli geliştiriyorum",
                "Veri analizi yapacağım",
                "Henüz ne yapacağımı bilmiyorum"
            ]
        
        # Project type discovery with diversity
        elif not context.get('project_type'):
            if diversity_mode:
                responses = [
                    "Bu kez farklı bir açıdan bakalım! Proje hedeflerinizi daha detaylı anlayabilir miyim? Her projenin kendine özgü bir hikayesi var ve sizinkini merak ediyorum. \n\nBu sefer hangi tür bir veri macerasına atılıyorsunuz? Belki hiç düşünmediğiniz bir yaklaşım keşfederiz!",
                    
                    "Yeni bir perspektifle yaklaşalım! Verilerinizle nasıl bir sonuca ulaşmak istiyorsunuz? Bu kez farklı algoritma ailelerini keşfetmek için sabırsızlanıyorum. \n\nProjenizin ana amacı nedir? Hangi tür çıktı elde etmeyi hedefliyorsunuz?",
                    
                    "Bu sefer bambaşka bir yoldan gidelim! Projenizin özünü anlayabilir miyim? Her seferinde farklı çözüm yolları keşfediyoruz. \n\nBu kez hangi tür analiz yapmayı planlıyorsunuz? Belki daha önce hiç düşünmediğiniz bir algoritma kategorisi bulabiliriz!"
                ]
            else:
                responses = [
                    "Vay, bu gerçekten ilginç geliyor! Benim deneyimime göre, projenin hedefini net anlamak algoritma seçiminin yarısı demek. \n\nMesela, geçen ay bir e-ticaret şirketi ile çalıştım - onlar müşteri davranışlarını tahmin etmek istiyordu. Sizin durumunuz nasıl? Hangi tür bir sonuç elde etmeyi hedefliyorsunuz? Verilerinizle ne yapmak istiyorsunuz?",
                    
                    "Şöyle düşünelim: Makine öğrenmesi biraz dedektiflik gibi - verilerden ipuçları toplayıp bir sonuca varıyoruz. Peki sizin 'gizeminiz' nedir? \n\nVerilerinizle şunlardan hangisini yapmak istiyorsunuz: Bir şeyleri kategorilere ayırmak mı, gelecekteki değerleri tahmin etmek mi, yoksa veriler arasındaki gizli kalıpları keşfetmek mi?"
                ]
            
            response = random.choice(responses)
            suggestions = [
                "Verileri kategorilere ayırmak istiyorum",
                "Gelecekteki değerleri tahmin etmek istiyorum",
                "Veri gruplarını keşfetmek istiyorum",
                "Anormal durumları tespit etmek istiyorum"
            ]
        
        # Data size discovery with context awareness
        elif not context.get('data_size'):
            project_type = context.get('project_type', 'proje')
            
            # Check conversation history for previous mentions
            conversation_context = self._get_conversation_context()
            
            if diversity_mode:
                responses = [
                    f"Şimdi {project_type} projeniz için veri boyutunu konuşalım! Bu kez farklı bir açıdan yaklaşmak istiyorum. Veri boyutu algoritma performansını doğrudan etkiler. \n\nBu sefer veri setinizin boyutu hakkında ne söyleyebilirsiniz? Kaç kayıt var yaklaşık olarak?",
                    
                    f"Bu kez {project_type} projenizin veri boyutunu keşfedelim! Her algoritmanın farklı veri boyutlarında farklı performans gösterdiğini biliyorsunuz. \n\nBu sefer veri setinizin büyüklüğü nasıl? Hangi aralıkta?"
                ]
            else:
                responses = [
                    f"Harika! {project_type} gerçekten güzel bir alan. Benim deneyimime göre, veri boyutu algoritma seçiminde kritik rol oynuyor. \n\nMesela, küçük veri setlerinde basit algoritmalar mucizeler yaratabilirken, büyük verilerde daha sofistike yaklaşımlar gerekiyor. Sizin veri setiniz hangi boyutta? Kaç kayıt var yaklaşık olarak?",
                    
                    f"Şöyle ki, {project_type} projesi için veri boyutu biraz yemeğin porsiyon miktarı gibi - az olursa farklı pişirme teknikleri, çok olursa farklı yaklaşımlar gerekiyor. \n\nVerilerinizin boyutu nasıl? Bu bilgi sayesinde size en verimli algoritmaları önerebilirim."
                ]
            
            response = random.choice(responses)
            suggestions = [
                "Küçük veri setim var (1000'den az)",
                "Orta boyut veri setim var (1000-10000)",
                "Büyük veri setim var (10000+)",
                "Çok büyük veri setim var (100000+)"
            ]
        
        # Data type discovery
        elif not context.get('data_type'):
            responses = [
                "Mükemmel! Veri boyutunu bilmek çok yardımcı oldu. Şimdi veri türünü öğrenmek istiyorum çünkü bu da algoritma seçimini doğrudan etkiliyor.\n\nBenim deneyimime göre, sayısal veriler farklı, metin verileri farklı yaklaşımlar istiyor. Tıpkı farklı dilleri konuşmak gibi - her biri kendine özgü kuralları var. Sizin verileriniz hangi türde?",
                
                "Şöyle düşünelim: Veriler biraz farklı dillerde yazılmış kitaplar gibi. Sayısal veriler matematik dili, metin verileri edebiyat dili, görüntüler ise sanat dili konuşuyor. \n\nSizin verileriniz hangi 'dilde' konuşuyor? Bu bilgi ile size en uygun 'çevirmen' algoritmayı bulabilirim."
            ]
            
            response = random.choice(responses)
            suggestions = [
                "Sayısal verilerle çalışıyorum",
                "Metin verileri işliyorum",
                "Kategorik verilerim var",
                "Görüntü verileri kullanıyorum"
            ]
        
        # Ready for recommendations
        else:
            response = "Harika! Artık projeniz hakkında yeterli bilgiye sahibim. Veri setinizin özelliklerini ve hedeflerinizi anlayarak size özel algoritma önerilerimi hazırlıyorum. \n\nBenim deneyimime göre, sizin durumunuz için birkaç mükemmel seçenek var. Hemen en uygun algoritmaları analiz edeyim!"
            suggestions = ["Algoritma önerilerini göster"]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "conversation_stage": self._get_conversation_stage(context)
        }

    def _summarize_conversation(self) -> str:
        """Summarize recent conversation for context"""
        if not self.conversation_memory:
            return "Yeni konuşma başlıyor"
        
        recent_topics = []
        for msg in self.conversation_memory[-5:]:
            if 'sınıflandırma' in msg['content'].lower():
                recent_topics.append('sınıflandırma')
            elif 'regresyon' in msg['content'].lower():
                recent_topics.append('regresyon')
            elif 'kümeleme' in msg['content'].lower():
                recent_topics.append('kümeleme')
        
        if recent_topics:
            return f"Son konuşulan konular: {', '.join(set(recent_topics))}"
        return "Genel algoritma danışmanlığı"

    def _summarize_user_profile(self) -> str:
        """Summarize user profile for context"""
        profile_parts = []
        
        if self.user_profile['experience_level'] != 'unknown':
            profile_parts.append(f"Deneyim: {self.user_profile['experience_level']}")
        
        if self.user_profile['preferred_style'] != 'unknown':
            profile_parts.append(f"Tercih: {self.user_profile['preferred_style']}")
        
        return ', '.join(profile_parts) if profile_parts else "Profil henüz belirlenmedi"

    def _generate_natural_suggestions(self, context: Dict, user_message: str) -> List[str]:
        """Generate natural, contextual suggestions"""
        if not context.get('project_type'):
            return [
                "Müşteri davranışlarını tahmin etmek istiyorum",
                "E-posta spam tespiti yapacağım",
                "Satış tahminleri yapmak istiyorum",
                "Görüntü tanıma projesi geliştiriyorum"
            ]
        elif not context.get('data_size'):
            return [
                "Birkaç yüz kayıt var",
                "Binlerce kayıt var",
                "On binlerce kayıt var",
                "Milyonlarca kayıt var"
            ]
        elif not context.get('data_type'):
            return [
                "Excel tablosunda sayısal veriler",
                "Müşteri yorumları ve metinler",
                "Ürün kategorileri ve etiketler",
                "Fotoğraf ve görüntü dosyaları"
            ]
        else:
            return [
                "En iyi algoritmaları öner",
                "Performans karşılaştırması yap",
                "Kod örnekleri ver",
                "Hangi metriği kullanmalıyım?"
            ]

    def _get_conversation_context(self) -> str:
        """Get conversation context from memory"""
        if not self.conversation_memory:
            return "Yeni konuşma"
        
        # Build context from conversation memory
        context_parts = []
        
        # Check for discussed algorithms
        if self.conversation_context['discussed_algorithms']:
            context_parts.append(f"Daha önce {', '.join(self.conversation_context['discussed_algorithms'])} algoritmalarını konuştuk.")
        
        # Check for user selections
        if self.conversation_context['user_selections']:
            last_selection = self.conversation_context['user_selections'][-1]
            context_parts.append(f"Özellikle {last_selection['algorithm']} algoritmasını tercih ettiğinizi belirttiniz.")
        
        # Check for recent feedback
        if self.conversation_context['user_feedback']:
            recent_feedback = self.conversation_context['user_feedback'][-1]
            if recent_feedback['type'] == 'positive':
                context_parts.append("Son önerilerimizi beğendiğinizi söylemiştiniz.")
            elif recent_feedback['type'] == 'negative':
                context_parts.append("Son önerilerimden memnun olmadığınızı belirttiniz.")
        
        # Check for last recommendations
        if self.conversation_context['last_recommendations']:
            rec_count = len(self.conversation_context['last_recommendations'])
            context_parts.append(f"Size {rec_count} algoritma önerisi sunmuştum.")
        
        return " ".join(context_parts) if context_parts else "Konuşmamız devam ediyor."
        
        # Analyze recent conversation
        recent_topics = []
        for msg in self.conversation_memory[-5:]:
            content = msg.get('content', '').lower()
            if 'sınıflandırma' in content or 'classification' in content:
                recent_topics.append('sınıflandırma')
            elif 'regresyon' in content or 'regression' in content:
                recent_topics.append('regresyon')
            elif 'kümeleme' in content or 'clustering' in content:
                recent_topics.append('kümeleme')
            elif 'algoritma' in content:
                recent_topics.append('algoritma')
        
        if recent_topics:
            return f"Önceki konuşma: {', '.join(set(recent_topics))}"
        return "Genel makine öğrenmesi"

    def _get_conversation_stage(self, context: Dict) -> str:
        """Determine current conversation stage"""
        if not context.get('project_type'):
            return 'project_discovery'
        elif not context.get('data_size'):
            return 'data_sizing'
        elif not context.get('data_type'):
            return 'data_typing'
        else:
            return 'recommendation_ready'

    def _generate_enhanced_recommendations(self, user_message: str, context: Dict) -> Dict:
        """Generate enhanced algorithm recommendations with natural language"""
        if self.openai_enabled:
            return self._generate_gpt4_enhanced_recommendations(user_message, context)
        else:
            return self._generate_template_enhanced_recommendations(user_message, context)

    def _generate_gpt4_enhanced_recommendations(self, user_message: str, context: Dict) -> Dict:
        """Generate recommendations using GPT-4 with natural storytelling"""
        # Get algorithm recommendations from our model
        recommendations = self.algorithm_recommender.get_recommendations(
            project_type=context.get('project_type'),
            data_size=context.get('data_size'),
            data_type=context.get('data_type'),
            complexity_preference='medium',
            top_n=3
        )
        
        # Prepare context for GPT-4
        recommendations_summary = []
        for rec in recommendations:
            confidence = rec.get('confidence_score', rec.get('confidence', 0.8))
            explanation = rec.get('explanation', rec.get('description', ''))
            recommendations_summary.append(f"- {rec['algorithm']} (Güven: {confidence:.1f}): {explanation}")
        
        context_prompt = f"""
Kullanıcının Proje Bilgileri:
- Proje türü: {context.get('project_type')}
- Veri boyutu: {context.get('data_size')}
- Veri türü: {context.get('data_type')}

Önerilen Algoritmalar:
{chr(10).join(recommendations_summary)}

Kullanıcının Son Mesajı: "{user_message}"

Görevin:
1. Algoritmaları hikaye anlatır gibi tanıt
2. Her algoritmanın "karakterini" ve "kişiliğini" açıkla
3. Gerçek dünya örnekleri ver
4. Hangi durumda hangisini seçeceğini açıkla
5. Kişisel deneyimlerini paylaş
6. Cesaretlendirici ve destekleyici ol
7. 3-4 paragraf halinde akıcı bir anlatım yap

Robotik listeler yerine hikaye anlatır gibi konuş!
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.algorithm_expert_prompt},
                    {"role": "user", "content": context_prompt}
                ],
                max_tokens=800,
                temperature=0.8,
                presence_penalty=0.2,
                frequency_penalty=0.2
            )
            
            gpt_response = response.choices[0].message.content
            
            # Generate follow-up suggestions
            suggestions = [
                f"{recommendations[0]['algorithm']} hakkında daha fazla bilgi",
                "Kod örnekleri göster",
                "Performans karşılaştırması yap",
                "Hangi metriği kullanmalıyım?"
            ]
            
            return {
                "response": gpt_response,
                "suggestions": suggestions,
                "recommendations": recommendations,
                "success": True,
                "ai_powered": True
            }
            
        except Exception as e:
            logger.error(f"GPT-4 recommendations failed: {e}")
            return self._generate_template_enhanced_recommendations(user_message, context)

    def _generate_template_enhanced_recommendations(self, user_message: str, context: Dict) -> Dict:
        """Generate enhanced recommendations using creative templates"""
        # Get algorithm recommendations
        recommendations = self.algorithm_recommender.get_recommendations(
            project_type=context.get('project_type'),
            data_size=context.get('data_size'),
            data_type=context.get('data_type'),
            complexity_preference='medium',
            top_n=3
        )
        
        if not recommendations:
            return self._get_emergency_fallback()
    
        # Create storytelling response
        project_type = context.get('project_type', 'proje')
        top_algo = recommendations[0]
        
        storytelling_intros = [
            f"Harika! {project_type} projeniz için analiz yaptım ve gerçekten heyecan verici sonuçlar çıktı. Benim deneyimime göre, sizin durumunuz için birkaç 'süper kahraman' algoritma var.",
            
            f"Vay canına! {project_type} projesi için mükemmel bir kombinasyon buldum. Şöyle ki, her algoritmanın kendine özgü bir 'kişiliği' var ve sizin verilerinizle harika bir uyum sağlayacak olanları seçtim.",
            
            f"Müjde! {project_type} alanında çok başarılı sonuçlar veren algoritmalar var ve sizin veri setiniz için özel olarak en uygun olanları analiz ettim."
        ]
        
        response = random.choice(storytelling_intros) + "\n\n"
        
        # Describe top algorithm with personality
        algo_personalities = {
            'Random Forest': "Random Forest gerçek bir 'takım oyuncusu' - yüzlerce küçük karar ağacından oluşan bir orkestra gibi çalışıyor. Benim deneyimime göre, çok güvenilir ve hatalarını kendi kendine düzelten nadir algoritmalardan biri.",
            
            'XGBoost': "XGBoost ise 'mükemmeliyetçi' bir karakter - her hatadan öğrenen ve sürekli kendini geliştiren bir algoritma. Kaggle yarışmalarının kralı diye boşuna demiyorlar!",
            
            'Logistic Regression': "Logistic Regression 'sade ve etkili' bir yaklaşım - bazen en basit çözümler en güçlü olanlar oluyor. Hızlı, anlaşılır ve güvenilir.",
            
            'K-Means': "K-Means 'organizatör' bir algoritma - karmaşık veri yığınlarını düzenli gruplara ayırmada uzman. Basit ama çok etkili.",
            
            'SVM': "SVM 'mükemmel sınır çizici' - veriler arasında en optimal sınırları bulan, matematiksel olarak çok zarif bir algoritma."
        }
        
        top_algo_name = top_algo['algorithm']
        if top_algo_name in algo_personalities:
            response += algo_personalities[top_algo_name]
        else:
            explanation = top_algo.get('explanation', top_algo.get('description', 'çok uygun bir seçim'))
            response += f"{top_algo_name} sizin projeniz için mükemmel bir seçim çünkü {explanation.lower()}"
        
        confidence = top_algo.get('confidence_score', top_algo.get('confidence', 0.8))
        response += f" Güven oranı %{confidence * 100:.0f} - bu gerçekten yüksek bir skor!\n\n"
        
        # Add practical advice
        practical_advice = [
            "Benim tavsiyem, önce bu algoritmayla başlayın ve sonuçları gözlemleyin. Genellikle ilk denemede çok iyi sonuçlar alıyorsunuz.",
            
            "Şöyle bir strateji öneriyorum: Bu algoritmayla temel modelinizi kurun, sonra diğer seçenekleri de deneyin ve karşılaştırın.",
            
            "Pratik açıdan bakarsak, bu algoritma sizin veri setinizle harika çalışacak. İsterseniz adım adım nasıl uygulayacağınızı da anlatabilirim."
        ]
        
        response += random.choice(practical_advice)
        
        # Generate contextual suggestions
        suggestions = [
            f"{top_algo_name} nasıl çalışır?",
            "Kod örneği ver",
            "Diğer algoritmaları da göster",
            "Performans karşılaştırması yap"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "recommendations": recommendations,
            "success": True
        }
    
    def _extract_enhanced_project_context(self, current_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Extract project context using AI analysis
        """
        # Handle None or empty messages
        if current_message is None:
            current_message = ""
            
        context = {
            'project_type': None,
            'data_size': None,
            'data_type': None,
            'class_count': None,
            'use_case': None,
            'constraints': [],
            'mentioned_algorithms': [],
            'conversation_stage': 'information_gathering'
        }
        
        # Combine all conversation content
        full_conversation = ""
        if conversation_history:
            for msg in conversation_history[-10:]:  # Last 10 messages
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    full_conversation += f"{role}: {content}\n"
        
        full_conversation += f"user: {current_message}"
        
        # Enhanced pattern matching with AI-like context understanding
        text_lower = full_conversation.lower()
        
        # Project type detection (more sophisticated)
        if any(word in text_lower for word in ['sınıflandırma', 'classification', 'kategorilere ayır', 'sınıflama', 'tahmin et', 'predict class']):
            context['project_type'] = 'classification'
        elif any(word in text_lower for word in ['kümeleme', 'clustering', 'segmentasyon', 'gruplama', 'segment']):
            context['project_type'] = 'clustering'
        elif any(word in text_lower for word in ['regresyon', 'regression', 'değer tahmin', 'fiyat tahmin', 'forecast']):
            context['project_type'] = 'regression'
        elif any(word in text_lower for word in ['anomali', 'outlier', 'anormal', 'dolandırıcılık', 'fraud']):
            context['project_type'] = 'anomaly_detection'
        elif any(word in text_lower for word in ['öneri', 'recommendation', 'tavsiye', 'suggest']):
            context['project_type'] = 'recommendation'
        
        # Data type detection (more intelligent defaults)
        if any(word in text_lower for word in ['sayısal', 'numerical', 'numeric', 'number', 'regresyon', 'regression']):
            context['data_type'] = 'numerical'
        elif any(word in text_lower for word in ['kategorik', 'categorical', 'category']):
            context['data_type'] = 'categorical'
        elif any(word in text_lower for word in ['metin', 'text', 'kelime', 'word']):
            context['data_type'] = 'text'
        elif any(word in text_lower for word in ['görüntü', 'image', 'resim', 'photo']):
            context['data_type'] = 'image'
        elif context.get('project_type'):
            # Default to numerical if project type is known but data type isn't specified
            context['data_type'] = 'numerical'
            
        # Data size defaults (more intelligent guessing)
        import re
        numbers = re.findall(r'\d+', text_lower)
        size_detected = False
        for num in numbers:
            num_val = int(num)
            if num_val < 1000:
                context['data_size'] = 'small'
                size_detected = True
                break
            elif num_val < 10000:
                context['data_size'] = 'medium'
                size_detected = True
                break
            else:
                context['data_size'] = 'large'
                size_detected = True
                break
        
        # Default data size if not detected but project type is known        
        if not size_detected and context.get('project_type'):
            context['data_size'] = 'medium'  # Safe default
                
        # Class count for classification
        if context['project_type'] == 'classification':
            if any(word in text_lower for word in ['2 sınıf', 'binary', 'ikili', 'two class']):
                context['class_count'] = 'binary'
            elif any(word in text_lower for word in ['3', '4', '5', 'few', 'az sınıf']):
                context['class_count'] = 'multiclass'
            elif any(word in text_lower for word in ['çok sınıf', 'many class', 'multiple']):
                context['class_count'] = 'multilabel'
            else:
                # Default to multiclass if classification is detected but no specific count given
                context['class_count'] = 'multiclass'
        
        # Extract mentioned algorithms
        algorithms = ['xgboost', 'random forest', 'svm', 'neural network', 'logistic regression', 'naive bayes', 'knn', 'decision tree']
        for algo in algorithms:
            if algo in text_lower:
                context['mentioned_algorithms'].append(algo)
        
        return context
    
    def _is_algorithm_specific_question(self, user_message: str, context: Dict) -> bool:
        """
        Check if user is asking specific questions about algorithms
        """
        user_msg_lower = user_message.lower()
        
        # Specific algorithm names that should always be handled regardless of context
        specific_algorithms = [
            'xgboost', 'random forest', 'svm', 'neural network', 'logistic regression',
            'holt-winters', 'linear regression', 'decision tree', 'naive bayes',
            'k-means', 'dbscan', 'pca', 'lstm', 'cnn', 'bert', 'transformer'
        ]
        
        # Algorithm-specific keywords that should bypass consultation
        algorithm_keywords = [
            'nasıl uygulanır', 'karşılaştır', 'kod örneği', 'performans', 
            'implementasyon', 'hangi algoritma', 'detay', 'açıkla',
            'örnek göster', 'nasıl yapılır', 'kıyasla', 'comparison', 'compare',
            'nasıl çalışır', 'avantaj', 'dezavantaj', 'ne zaman kullan',
            'performans karşılaştır', 'algoritma karşılaştır', 'hangisi daha iyi'
        ]
        
        # Check for specific algorithm names (these are handled regardless of context)
        contains_specific_algo = any(algo in user_msg_lower for algo in specific_algorithms)
        
        # Check for algorithm-specific keywords
        contains_algo_keyword = any(keyword in user_msg_lower for keyword in algorithm_keywords)
        
        # If user mentions specific algorithm, handle immediately
        if contains_specific_algo:
            return True
            
        # For general algorithm questions, we're more flexible now
        # Performance comparison and code examples should always be handled
        if contains_algo_keyword:
            # These specific types should always be handled
            if any(word in user_msg_lower for word in ['performans', 'karşılaştır', 'kod örneği', 'nasıl uygulanır']):
                return True
            # For other algorithm keywords, require some context
            has_minimum_info = context.get('project_type') is not None
            return has_minimum_info
        
        return False
    
    def _handle_algorithm_question(self, user_message: str, context: Dict) -> Dict:
        """
        Handle specific algorithm questions without restarting consultation
        """
        user_msg_lower = user_message.lower()
        
        # Code example requests
        if 'kod örneği' in user_msg_lower or 'nasıl uygulanır' in user_msg_lower:
            return self._generate_code_example(user_message, context)
        
        # Performance comparison requests
        elif 'performans' in user_msg_lower or 'karşılaştır' in user_msg_lower:
            return self._generate_performance_comparison(context)
        
        # Algorithm explanation requests - expanded keywords
        elif any(word in user_msg_lower for word in ['detay', 'açıkla', 'nedir', 'nasıl çalışır', 'ne yapar', 'avantaj', 'dezavantaj', 'ne zaman kullan', 'bilgi', 'hakkında', 'anlat', 'öğren']):
            return self._generate_algorithm_explanation(user_message, context)
        
        # Default: ask for clarification
        else:
            return {
                "response": "🤔 Hangi algoritma hakkında bilgi almak istiyorsunuz?\n\nPopüler seçenekler:\n• K-means\n• Random Forest\n• XGBoost\n• SVM\n• Neural Networks\n\nHangisi hakkında detay istiyorsunuz?",
                "suggestions": [
                    "K-means hakkında bilgi ver",
                    "Random Forest nasıl çalışır?",
                    "XGBoost algoritması nedir?",
                    "SVM açıkla"
                ],
                "success": True
            }
    
    def _generate_code_example(self, user_message: str, context: Dict) -> Dict:
        """
        Generate code examples using GPT-4 or enhanced fallback system
        """
        # Try GPT-4 first for personalized code examples
        if self.openai_enabled and self.openai_client:
            try:
                return self._generate_gpt4_code_example(user_message, context)
            except Exception as e:
                print(f"⚠️ GPT-4 code generation failed, using template: {e}")
        
        # Fallback to template system
        return self._generate_template_code_example(user_message, context)
    
    def _generate_gpt4_code_example(self, user_message: str, context: Dict) -> Dict:
        """
        Generate personalized code examples using GPT-4
        """
        project_type = context.get('project_type', 'classification')
        data_size = context.get('data_size', 'medium')
        
        # Detect which algorithm user is asking about
        user_msg_lower = user_message.lower()
        algorithm = "genel"
        
        for algo in ['xgboost', 'random forest', 'svm', 'neural network', 'logistic regression', 'holt-winters', 'linear regression']:
            if algo in user_msg_lower:
                algorithm = algo
                break
        
        context_info = f"""
Proje türü: {project_type}
Veri boyutu: {data_size}
Veri türü: {context.get('data_type', 'numerical')}
İstenilen algoritma: {algorithm}
Kullanıcı sorusu: "{user_message}"
"""
        
        prompt = f"""
Sen senior-level bir makine öğrenmesi uzmanı ve Python geliştiricisisin. Kullanıcıya industry-standard, production-ready kod örnekleri sunuyorsun.

{context_info}

Lütfen profesyonel bir danışman gibi:

📋 **Kod Kalitesi:**
- Clean, readable ve well-documented Python kodu yaz
- Best practices ve design patterns kullan
- Error handling ve edge case'leri dahil et
- Type hints ve docstring'ler ekle

🎯 **Algoritma Seçimi:**
- Projenin gereksinimlerine göre en optimal algoritmaları öner
- Hyperparameter tuning stratejileri sun
- Performance optimization ipuçları ver
- Cross-validation ve model evaluation detayları ekle

💡 **Açıklamalar:**
- Teknik detayları paragraf halinde açıkla
- Algoritmanın çalışma prensiplerini anlat
- Ne zaman hangi algoritmanın kullanılacağını belirt
- Production environment için deployment ipuçları ver

🚀 **Professional Touch:**
- Industry best practices dahil et
- Scalability ve maintainability dikkate al
- Memory ve computational efficiency önerileri sun
- Real-world kullanım senaryolarını anlat

Yanıtın hem teknik derinlikte hem de kolayca uygulanabilir olsun. Senior developer seviyesinde kod ve açıklama bekliyorum.
"""
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Bu proje için {algorithm} algoritmasının Python implementasyonunu ve açıklamasını verir misin?"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use the latest GPT-3.5 Turbo model for best quality
            messages=messages,
            max_tokens=1500,  # Increased for more detailed responses
            temperature=0.2   # Lower temperature for more consistent professional responses
        )
        
        gpt_response = response.choices[0].message.content
        
        suggestions = [
            "Hiperparametre optimizasyonu",
            "Cross-validation ekleme",
            "Feature engineering ipuçları",
            "Başka algoritma kodu"
        ]
        
        return {
            "response": gpt_response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": True
        }
    
    def _generate_template_code_example(self, user_message: str, context: Dict) -> Dict:
        """
        Enhanced template-based code examples with detailed explanations
        """
        project_type = context.get('project_type', 'classification')
        
        if 'xgboost' in user_message.lower() or 'random forest' in user_message.lower():
            algo_name = 'XGBoost' if 'xgboost' in user_message.lower() else 'Random Forest'
            
            if project_type == 'classification':
                code_example = f"""**{algo_name} ile Sınıflandırma - Detaylı Uygulama:**

Bu algoritma {context.get('data_size', 'orta')} boyuttaki veri setiniz için mükemmel bir seçim. Hem yüksek performans hem de güvenilirlik sunar.

```python
# Gerekli kütüphaneleri yükleme
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Veri yükleme ve ilk inceleme
df = pd.read_csv('your_data.csv')
print(f"Veri seti boyutu: {{df.shape}}")
print(f"Eksik değer sayısı: {{df.isnull().sum().sum()}}")

# Özellik ve hedef değişkenleri ayırma
X = df.drop('target_column', axis=1)  # Hedef sütununuzun adını yazın
y = df['target_column']

# Veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Random Forest modelini oluşturma ve eğitme
# n_estimators: Ağaç sayısı (daha fazla = daha iyi performans ama yavaş)
# max_depth: Ağaçların maksimum derinliği (overfitting'i kontrol eder)
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1  # Tüm CPU'ları kullan
)

# Modeli eğitme
print("Model eğitiliyor...")
rf_model.fit(X_train, y_train)

# Tahmin yapma
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Performans değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk oranı: {{accuracy:.3f}}")

# Detaylı performans raporu
print("\\nDetaylı Performans Raporu:")
print(classification_report(y_test, y_pred))

# Cross-validation ile daha güvenilir performans ölçümü
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print(f"\\n5-Fold CV Ortalama Doğruluk: {{cv_scores.mean():.3f}} (+/- {{cv_scores.std()*2:.3f}})")

# Özellik önemlerini görüntüleme
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\nEn önemli özellikler:")
print(feature_importance.head(10))
```

**Önemli İpuçları:**

Veri setinizin boyutuna göre parametreleri ayarlayın. Küçük veri setlerde n_estimators=50-100 yeterli, büyük veri setlerde 200-500 arası deneyebilirsiniz. max_depth parametresi overfitting'i kontrol eder - başlangıç için 10-15 arasında deneyin.

Model eğitildikten sonra feature_importance değerleriyle hangi özelliklerin en çok etkili olduğunu görebilirsiniz. Bu size veri anlama konusunda büyük insight verir."""
            else:
                code_example = f"""**{algo_name} ile Regresyon - Kapsamlı Uygulama:**

Sayısal tahmin problemleriniz için {algo_name} mükemmel bir seçim. Özellikle {context.get('data_size', 'orta')} boyuttaki veri setlerde çok başarılı.

```python
# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Veri hazırlama
df = pd.read_csv('your_regression_data.csv')
X = df.drop('target_column', axis=1)
y = df['target_column']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model oluşturma - regresyon için optimize edilmiş parametreler
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Model eğitimi
rf_regressor.fit(X_train, y_train)

# Tahminler
y_pred = rf_regressor.predict(X_test)

# Performans metrikleri
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performansı:")
print(f"R² Score: {{r2:.3f}}")
print(f"RMSE: {{rmse:.3f}}")
print(f"MAE: {{mae:.3f}}")

# Tahmin vs Gerçek değerler grafiği
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Tahmin vs Gerçek Değerler')
plt.show()
```

Bu kod size hem model performansını hem de tahminlerin görsel analizini sağlar. R² değeri 0.8'in üstündeyse modeliniz çok başarılı demektir."""
        else:
            code_example = f"""**Genel Machine Learning Pipeline - {project_type.title()} için:**

Projeniz için kapsamlı bir başlangıç şablonu hazırladım. Bu kod yapısını temel alarak istediğiniz algoritmaları deneyebilirsiniz.

```python
# Temel kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Veri Yükleme ve İnceleme
print("=== VERİ ANALİZİ ===")
df = pd.read_csv('your_data.csv')
print(f"Veri boyutu: {{df.shape}}")
print(f"Sütunlar: {{list(df.columns)}}")
print(f"\\nVeri tipleri:\\n{{df.dtypes}}")
print(f"\\nEksik değerler:\\n{{df.isnull().sum()}}")

# 2. Veri Ön İşleme
print("\\n=== VERİ ÖN İŞLEME ===")

# Kategorik değişkenleri encode etme
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    if col != 'target_column':  # Hedef değişken değilse
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Özellik ve hedef ayırma
X = df.drop('target_column', axis=1)
y = df['target_column']

# Verileri normalize etme (önemli!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model Seçimi ve Eğitimi
print("\\n=== MODEL EĞİTİMİ ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Buraya istediğiniz algoritmanın kodunu ekleyebilirsiniz
# Örnek: RandomForestClassifier, SVM, XGBoost vb.

print("Model başarıyla eğitildi!")
print("Şimdi istediğiniz algoritma kodunu ekleyebilirsiniz.")
```

Bu temel yapıyı kullanarak istediğiniz algoritmanın detaylı kodunu sorabilirsiniz. Hangi algoritma ile devam etmek istersiniz?"""
        
        return {
            "response": code_example,
            "suggestions": [
                "XGBoost kodu",
                "SVM implementasyonu",
                "Hiperparametre optimizasyonu",
                "Cross-validation ekleme"
            ],
            "success": True
        }
    
    def _generate_performance_comparison(self, context: Dict) -> Dict:
        """
        Generate performance comparison between algorithms
        """
        project_type = context.get('project_type', 'classification')
        data_size = context.get('data_size', 'medium')
        
        if project_type == 'classification':
            comparison = """Sınıflandırma Algoritmaları Karşılaştırması:

1. Logistic Regression
   - Doğruluk: Orta
   - Hız: Çok hızlı
   - Anlaşılabilirlik: Çok kolay
   - En iyi: Küçük veri setleri

2. Random Forest
   - Doğruluk: İyi
   - Hız: Orta
   - Anlaşılabilirlik: Kolay
   - En iyi: Genel kullanım

3. XGBoost
   - Doğruluk: Çok iyi
   - Hız: Orta
   - Anlaşılabilirlik: Zor
   - En iyi: Büyük veri setleri

4. SVM
   - Doğruluk: İyi
   - Hız: Yavaş
   - Anlaşılabilirlik: Zor
   - En iyi: Küçük, karmaşık veriler

"""
            if data_size == 'small':
                comparison += "Küçük veri setiniz için: Logistic Regression veya SVM önerilir."
            elif data_size == 'large':
                comparison += "Büyük veri setiniz için: XGBoost veya Random Forest önerilir."
            else:
                comparison += "Genel kullanım için: Random Forest ile başlayın."
        else:
            comparison = """Regresyon Algoritmaları Karşılaştırması:

1. Linear Regression
   - Doğruluk: Orta
   - Hız: Çok hızlı
   - Anlaşılabilirlik: Çok kolay

2. Random Forest
   - Doğruluk: İyi
   - Hız: Orta
   - Anlaşılabilirlik: Kolay

3. XGBoost
   - Doğruluk: Çok iyi
   - Hız: Orta
   - Anlaşılabilirlik: Zor

En iyi seçim veri setinizin boyutuna bağlıdır."""
        
        return {
            "response": comparison,
            "suggestions": [
                "Hangi metrik kullanmalıyım?",
                "Cross-validation nasıl yapılır?",
                "Hiperparametre optimizasyonu"
            ],
            "success": True
        }
    
    def _generate_algorithm_explanation(self, user_message: str, context: Dict) -> Dict:
        """
        Generate detailed explanation about specific algorithms
        """
        explanations = {
            'xgboost': """
🏆 **XGBoost (Extreme Gradient Boosting):**

**Ne yapar?**
Zayıf öğrenicileri (karar ağaçları) sıralı olarak birleştirerek güçlü bir model oluşturur.

**Avantajları:**
✅ Çok yüksek doğruluk
✅ Eksik verilerle başa çıkabilir
✅ Özellik önemini gösterir
✅ Büyük veri setlerinde hızlı

**Dezavantajları:**
❌ Karmaşık hiperparametre ayarı
❌ Overfitting eğilimi
❌ Yorumlanması zor

**Ne zaman kullanmalı?**
• Maksimum performans istediğinizde
• Yarışmalarda (Kaggle'da çok popüler)
• Büyük ve karmaşık veri setlerinde
""",
            'random forest': """
🌳 **Random Forest:**

**Ne yapar?**
Birçok karar ağacını aynı anda eğitir ve sonuçlarını birleştirir.

**Avantajları:**
✅ Overfitting'e dirençli
✅ Değişken önemini gösterir
✅ Eksik verilerle çalışabilir
✅ Hem classification hem regression

**Dezavantajları:**
❌ Büyük model boyutu
❌ Gerçek zamanlı tahminlerde yavaş olabilir

**Ne zaman kullanmalı?**
• Güvenilir bir başlangıç algoritması olarak
• Özellik önemini anlamak için
• Hem hız hem doğruluk istediğinizde
""",
            'k-means': """
🎯 **K-Means Clustering:**

**Ne yapar?**
Verileri önceden belirlenen sayıda (k) gruba böler. Her grup bir merkez etrafında toplanır.

**Avantajları:**
✅ Basit ve hızlı
✅ Büyük veri setlerinde etkili
✅ Yorumlanması kolay
✅ Bellek kullanımı düşük

**Dezavantajları:**
❌ K sayısını önceden belirlemelisiniz
❌ Küresel olmayan şekillerde zayıf
❌ Aykırı değerlere hassas
❌ Farklı boyutlardaki grupları ayırmada zor

**Ne zaman kullanmalı?**
• Müşteri segmentasyonu
• Veri ön işleme için
• Görüntü işlemede renk azaltma
• Pazarlama analizi

**Python Örneği:**
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Model oluşturma
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Sonuçları görselleştirme
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.scatter(kmeans.cluster_centers_[:, 0], 
           kmeans.cluster_centers_[:, 1], 
           marker='x', s=200, c='red')
plt.show()
```
""",
            'svm': """
⚡ **Support Vector Machine (SVM):**

**Ne yapar?**
Sınıflar arasında en geniş marjinli ayırıcı çizgiyi/düzlemi bulur.

**Avantajları:**
✅ Yüksek boyutlu verilerde etkili
✅ Bellek kullanımı verimli
✅ Çok çeşitli kernel fonksiyonları
✅ Overfitting'e dirençli

**Dezavantajları:**
❌ Büyük veri setlerinde yavaş
❌ Hiperparametre ayarı kritik
❌ Olasılık tahmini yapmaz
❌ Noise'a hassas

**Ne zaman kullanmalı?**
• Metin sınıflandırma
• Görüntü tanıma
• Yüksek boyutlu veriler
• Küçük-orta boyutlu veri setleri

**Python Örneği:**
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Model oluşturma
svm = SVC(kernel='rbf', random_state=42)

# Hiperparametre optimizasyonu
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# En iyi model
best_svm = grid_search.best_estimator_
```
""",
            'holt-winters': """
📈 **Holt-Winters (Triple Exponential Smoothing):**

**Ne yapar?**
Zaman serisi verilerindeki trend, sezonluk ve seviye bileşenlerini ayrı ayrı modelleyerek gelecek tahminleri yapar.

**Avantajları:**
✅ Sezonsal verilerde çok başarılı
✅ Trend ve mevsimsellik yakalar
✅ Yorumlanabilir sonuçlar
✅ Hesaplama açısından hızlı

**Dezavantajları:**
❌ Sadece zaman serisi verileri için
❌ Ani değişimlere karşı hassas
❌ Parametrelerin doğru ayarlanması gerekli

**Ne zaman kullanmalı?**
• Mevsimsel satış tahminleri
• Enerji tüketim projeksiyonları
• Düzenli döngüsel veriler
• Kısa-orta vadeli tahminler

**Python Örneği:**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Model oluşturma
model = ExponentialSmoothing(data, trend='add', seasonal='add')
fitted_model = model.fit()

# Tahmin yapma
forecast = fitted_model.forecast(steps=12)
```
"""
        }
        
        user_msg_lower = user_message.lower()
        
        # Check for algorithm mentions with better matching
        for algo, explanation in explanations.items():
            algo_variants = [algo, algo.replace('-', ''), algo.replace(' ', '')]
            if any(variant in user_msg_lower for variant in algo_variants):
                return {
                    "response": explanation,
                    "suggestions": [
                        f"{algo.title()} kod örneği",
                        "Hiperparametre ayarları",
                        "Diğer algoritmalarla karşılaştır"
                    ],
                    "success": True
                }
        
        # Generic algorithm explanation - check if any algorithm mentioned in message
        mentioned_algorithms = []
        for algo in ['xgboost', 'random forest', 'svm', 'neural network', 'logistic regression', 'k-means', 'kmeans', 'dbscan', 'optics']:
            if algo in user_msg_lower or algo.replace('-', '').replace(' ', '') in user_msg_lower.replace('-', '').replace(' ', ''):
                mentioned_algorithms.append(algo)
        
        if mentioned_algorithms:
            algo = mentioned_algorithms[0]
            return {
                "response": f"🤖 {algo.title()} hakkında bilgi istiyorsunuz. Size bu algoritmanın detaylarını açıklayabilirim. Hangi konuda daha fazla bilgi istiyorsunuz?",
                "suggestions": [
                    f"{algo.title()} nasıl çalışır?",
                    f"{algo.title()} avantajları neler?",
                    f"{algo.title()} kod örneği",
                    "Diğer algoritmalarla karşılaştır"
                ],
                "success": True
            }
        else:
            return {
                "response": "🤖 Hangi algoritma hakkında bilgi almak istiyorsunuz? Size detaylarını açıklayabilirim.",
                "suggestions": [
                    "XGBoost nedir?",
                    "Random Forest açıkla",
                    "SVM nasıl çalışır?",
                    "K-means hakkında bilgi ver"
                ],
                "success": True
            }
    
    def _should_recommend_algorithms(self, context: Dict) -> bool:
        """
        Determine if we have enough information to make algorithm recommendations
        """
        required_info = ['project_type', 'data_size', 'data_type']
        
        # For classification, we're more flexible about class_count
        # We can still give recommendations without it
        
        gathered_info = [key for key in required_info if context.get(key) is not None]
        
        print(f"📋 Required: {required_info}")
        print(f"📋 Gathered: {gathered_info}")
        
        # If we have project_type, we can give basic recommendations
        # Even more flexible: just 1 piece of info is enough for basic suggestions
        return len(gathered_info) >= 1
    
    def _generate_algorithm_recommendations(self, user_message: str, context: Dict) -> Dict:
        """
        Generate algorithm recommendations using our custom model + advanced AI explanation
        """
        try:
            # Use our algorithm recommender
            recommendations = self.algorithm_recommender.get_recommendations(
                project_type=context.get('project_type', 'classification'),
                data_size=context.get('data_size', 'medium'),
                data_type=context.get('data_type', 'numerical'),
                complexity_preference='medium'
            )
            
            # Always use our advanced AI system for better explanations
            return self._advanced_ai_recommendations(user_message, context, recommendations)
                
        except Exception as e:
            print(f"❌ Error in recommendations: {e}")
            return self._get_emergency_fallback()
    
    def _advanced_ai_recommendations(self, user_message: str, context: Dict, recommendations: List[Dict]) -> Dict:
        """
        Hybrid AI system: GPT-4 + Custom Algorithm Model for personalized recommendations
        """
        try:
            if not recommendations:
                return self._get_emergency_fallback()
            
            # Get top 3 recommendations
            top_algos = recommendations[:3]
            
            # Try GPT-4 first for detailed paragraph responses
            if self.openai_enabled and self.openai_client:
                try:
                    return self._generate_gpt4_recommendations(user_message, context, top_algos)
                except Exception as e:
                    print(f"⚠️ GPT-4 failed, using advanced fallback: {e}")
            
            # Fallback to enhanced template system
            return self._generate_enhanced_template_recommendations(user_message, context, top_algos)
                
        except Exception as e:
            print(f"❌ Advanced AI recommendation error: {e}")
            return self._template_recommendations(context, recommendations)
    
    def _generate_gpt4_recommendations(self, user_message: str, context: Dict, recommendations: List[Dict]) -> Dict:
        """
        Generate detailed paragraph recommendations using GPT-4
        """
        try:
            # Prepare algorithm details for GPT-4
            algo_details = []
            for algo in recommendations:
                algo_details.append(f"- {algo['algorithm']}: Güven skoru {algo['confidence_score']:.1f}/5.0")
            
            # Create context string
            project_info = f"""
Proje türü: {context.get('project_type', 'Belirsiz')}
Veri boyutu: {context.get('data_size', 'Belirsiz')}
Veri türü: {context.get('data_type', 'Belirsiz')}
Sınıf sayısı: {context.get('class_count', 'Belirsiz')}

Önerilen algoritmalar:
{chr(10).join(algo_details)}

Kullanıcı mesajı: "{user_message}"
"""
            
            # GPT-4 prompt
            messages = [
                {"role": "system", "content": self.algorithm_expert_prompt},
                {"role": "user", "content": f"""
Yukarıdaki proje bilgilerine dayanarak algoritma önerilerimi paragraf halinde detaylı açıkla.

{project_info}

Lütfen:
1. Her algoritmayı neden önerdiğimi paragraf halinde açıkla
2. Projenin özelliklerine göre avantajları belirt
3. Pratik uygulama ipuçları ver
4. Hangi algoritma ile başlanmasını öneriyorsan belirt
5. Samimi ve anlaşılır bir dille yaz

Kısa maddeler yerine akıcı paragraflar halinde cevap ver.
"""}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo for superior algorithmic advice and detailed explanations
                messages=messages,
                max_tokens=800,
                temperature=0.7
            )
            
            gpt_response = response.choices[0].message.content
            
            # Generate smart suggestions
            suggestions = []
            if recommendations:
                suggestions.append(f"{recommendations[0]['algorithm']} kod örneği")
                suggestions.append("Performans karşılaştırması yap")
                suggestions.append("Hiperparametre optimizasyonu")
                if len(recommendations) > 1:
                    suggestions.append(f"{recommendations[1]['algorithm']} detayları")
            
            return {
                "response": gpt_response,
                "suggestions": suggestions,
                "success": True,
                "ai_powered": True
            }
            
        except Exception as e:
            print(f"❌ GPT-4 recommendation error: {e}")
            raise e
    
    def _generate_enhanced_template_recommendations(self, user_message: str, context: Dict, recommendations: List[Dict]) -> Dict:
        """
        Enhanced fallback system with paragraph-style responses
        """
        project_type = context.get('project_type') or 'machine learning'
        data_size = context.get('data_size') or 'medium' 
        data_type = context.get('data_type') or 'numerical'
        
        # Track recommended algorithms
        for rec in recommendations:
            algo_name = rec.get('algorithm', '').lower()
            if algo_name and algo_name not in self.conversation_context['discussed_algorithms']:
                self.conversation_context['discussed_algorithms'].append(algo_name)
        
        # Create paragraph-style introduction
        if project_type == 'classification':
            intro = f"Sınıflandırma projeniz için detaylı analiz yaptım ve size en uygun algoritmaları seçtim. {data_size.title()} boyuttaki {data_type} veriniz için özellikle etkili olacak çözümler buldum."
        elif project_type == 'regression':
            intro = f"Regresyon analiziniz için algoritma seçiminde dikkat ettiğim temel faktörler veri boyutunuz ({data_size}) ve veri tipinizdir ({data_type}). Bu özelliklere göre en başarılı sonuçları verecek algoritmaları önceledim."
        else:
            intro = f"Projeniz için uygun algoritma seçiminde veri karakteristiklerinizi göz önünde bulundurdum. {data_size.title()} boyuttaki {data_type} verileriniz için optimize edilmiş önerilerimi paylaşıyorum."
        
        response = intro + "\n\n"
        
        # Detailed algorithm explanations in paragraph form
        for i, algo in enumerate(recommendations[:3], 1):
            algo_name = algo['algorithm']
            confidence = algo['confidence_score']
            
            if i == 1:
                response += f"**{algo_name}** algoritmasını ilk sırada öneriyorum çünkü {confidence:.1f}/5.0 güven skoru ile projenize en uygun seçenek. "
            else:
                response += f"**{algo_name}** da {confidence:.1f}/5.0 güven skoru ile güçlü bir alternatif. "
            
            # Get detailed explanation
            explanation = self._get_enhanced_explanation(algo_name, context)
            response += explanation + "\n\n"
        
        # Contextual advice paragraph
        if data_size == 'small':
            response += "Küçük veri setiniz göz önünde bulundurulduğunda, overfitting riskini minimize etmek için daha basit modelleri tercih etmenizi öneriyorum. Başlangıç için ilk önerdiğim algoritmayı deneyip sonuçları değerlendirdikten sonra diğer seçeneklere geçebilirsiniz."
        elif data_size == 'large':
            response += "Büyük veri setinizin avantajını kullanarak daha karmaşık modelleri güvenle deneyebilirsiniz. Bu durumda ensemble metotları ve derin öğrenme yaklaşımları özellikle etkili sonuçlar verebilir."
        else:
            response += "Orta boyuttaki veri setiniz için dengeli bir yaklaşım öneriyorum. İlk etapta daha basit algoritmalarla başlayıp performans sonuçlarına göre karmaşıklığı artırabilirsiniz."
        
        # Generate suggestions
        suggestions = [
            f"{recommendations[0]['algorithm']} nasıl uygulanır?",
            "Performans karşılaştırması",
            "Kod örnekleri ver",
            "Hangi metriği kullanmalıyım?"
        ]
        
        if len(recommendations) > 1:
            suggestions.append(f"{recommendations[1]['algorithm']} detayları")
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
    
    def _get_simple_explanation(self, algorithm: str, context: Dict) -> str:
        """
        Get simple, clean explanation for each algorithm
        """
        explanations = {
            'XGBoost': "Yüksek doğruluk oranına sahip, güçlü bir algoritma. Çoğu durumda çok iyi sonuçlar verir.",
            'Random Forest': "Güvenilir ve dengeli bir seçim. Overfitting yapmaz, sonuçları yorumlaması kolay.",
            'Logistic Regression': "Basit ve hızlı. Başlangıç için ideal, sonuçları anlaşılır.",
            'SVM': "Karmaşık veri ilişkilerini iyi yakalar. Küçük veri setlerinde başarılı.",
            'Neural Network': "Karmaşık problemleri çözebilir. Büyük veri setleri gerektir.",
            'Linear Regression': "Basit ve hızlı regresyon algoritması. Yorumlaması kolay.",
            'Decision Tree': "Anlaşılması kolay kural tabanlı algoritma.",
            'Naive Bayes': "Hızlı ve basit sınıflandırma algoritması.",
            'K-Means': "Veri gruplarını otomatik olarak bulur.",
            'DBSCAN': "Gürültülü verilerde grup bulma algoritması.",
        }
        
        return explanations.get(algorithm, "Güvenilir bir makine öğrenmesi algoritması.")
    
    def _get_algorithm_explanation(self, algorithm: str, context: Dict, confidence: float) -> str:
        """
        Get intelligent, contextual explanation for each algorithm
        """
        explanations = {
            'XGBoost': {
                'classification': "🏆 Gradient boosting'in şampiyonu! Karmaşık ilişkileri yakalama konusunda uzman. Kaggle yarışmalarının favorisi.",
                'regression': "📈 Sayısal tahminlerde çok güçlü! Eksik verilerle bile başarılı çalışır.",
                'general': "⚡ Hızlı, güçlü ve esnek. Çoğu problemde harika sonuçlar verir."
            },
            'Random Forest': {
                'classification': "🌳 Karar ağaçlarının gücünü birleştirir. Overfitting'e karşı dirençli ve yorumlanabilir.",
                'regression': "🌲 Stabil tahminler yapar. Özellik önemini gösterir.",
                'general': "🔒 Güvenilir ve robust. Hemen hemen her veri türüyle çalışır."
            },
            'Logistic Regression': {
                'classification': "📊 Basit ama etkili! İkili sınıflandırmada mükemmel. Sonuçları anlamak kolay.",
                'general': "✨ Hızlı ve yorumlanabilir. Başlangıç için ideal seçim."
            },
            'SVM': {
                'classification': "🎯 Karmaşık veri sınırlarını çizer. Yüksek boyutlu verilerde başarılı.",
                'general': "💪 Güçlü matematik temeli. Kernel trick ile sihir yapar."
            },
            'Neural Network': {
                'classification': "🧠 Beyin yapısını taklit eder. Çok karmaşık pattern'leri öğrenebilir.",
                'general': "🚀 Derin öğrenmenin kapısı. Büyük verilerle şaha kalkar."
            }
        }
        
        project_type = context.get('project_type', 'general')
        
        if algorithm in explanations:
            if project_type in explanations[algorithm]:
                base_explanation = explanations[algorithm][project_type]
            else:
                base_explanation = explanations[algorithm]['general']
        else:
            base_explanation = "🔧 Güvenilir bir algoritma. Projenizde iyi sonuçlar verebilir."
        
        # Add confidence-based comment
        if confidence >= 4.5:
            confidence_note = "✅ Size özel olarak optimize edilmiş!"
        elif confidence >= 4.0:
            confidence_note = "👍 Verilerinizle uyumlu!"
        elif confidence >= 3.5:
            confidence_note = "📝 Denemeye değer!"
        else:
            confidence_note = "🤔 Alternatif seçenek olabilir."
            
        return f"{base_explanation} {confidence_note}"
    
    def _generate_consultation_response(self, user_message: str, context: Dict) -> Dict:
        """
        Generate consultation questions using advanced AI to gather more information
        """
        # Always use our advanced AI system for better user experience
        return self._advanced_ai_consultation_response(user_message, context)
    
    def _advanced_ai_consultation_response(self, user_message: str, context: Dict) -> Dict:
        """
        Hybrid AI consultation: GPT-4 + template fallback for gathering project information
        """
        try:
            # Handle None or empty messages
            if user_message is None:
                user_message = ""
            
            # Try GPT-4 first for personalized consultation
            if self.openai_enabled and self.openai_client:
                try:
                    return self._generate_gpt4_consultation(user_message, context)
                except Exception as e:
                    print(f"⚠️ GPT-4 consultation failed, using template: {e}")
            
            # Fallback to enhanced template consultation
            return self._generate_template_consultation(user_message, context)
            
        except Exception as e:
            print(f"❌ Advanced AI consultation error: {e}")
            return self._template_consultation_response(context)
    
    def _generate_gpt4_consultation(self, user_message: str, context: Dict) -> Dict:
        """
        Generate personalized consultation using GPT-4
        """
        # Determine what information we still need
        missing_info = []
        if not context.get('project_type'):
            missing_info.append('proje türü')
        if not context.get('data_size'):
            missing_info.append('veri boyutu')
        if not context.get('data_type'):
            missing_info.append('veri türü')
        if context.get('project_type') == 'classification' and not context.get('class_count'):
            missing_info.append('sınıf sayısı')
        
        # Prepare context for GPT-4
        context_info = f"""
Mevcut proje bilgileri:
- Proje türü: {context.get('project_type', 'Henüz belirlenmedi')}
- Veri boyutu: {context.get('data_size', 'Henüz belirlenmedi')}
- Veri türü: {context.get('data_type', 'Henüz belirlenmedi')}
- Sınıf sayısı: {context.get('class_count', 'Henüz belirlenmedi')}

Eksik bilgiler: {', '.join(missing_info) if missing_info else 'Yok'}

Kullanıcı mesajı: "{user_message}"
"""
        
        messages = [
            {"role": "system", "content": self.consultation_prompt},
            {"role": "user", "content": f"""
Bir kullanıcı algoritma danışmanlığı için geldi. Aşağıdaki bilgileri göz önünde bulundurarak ona yardım et:

{context_info}

Lütfen:
1. Kullanıcının mesajına samimi ve paragraf halinde cevap ver
2. Eksik bilgileri nazikçe sor ama zorlama
3. Projenin hedefini net anlayıp doğru yönlendir
4. Teknik terimlerden kaçın, sade konuş
5. 2-3 paragraf halinde cevap ver

Kısa listeler yerine akıcı konuşma yap.
"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use GPT-3.5 Turbo for consultation responses
            messages=messages,
            max_tokens=400,
            temperature=0.8
        )
        
        gpt_response = response.choices[0].message.content
        
        # Generate contextual suggestions
        suggestions = self._generate_consultation_suggestions(missing_info, context)
        
        return {
            "response": gpt_response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": True
        }
    
    def _generate_template_consultation(self, user_message: str, context: Dict) -> Dict:
        """
        Enhanced template-based consultation with paragraph responses
        """
        # Determine what information we still need
        missing_info = []
        if not context.get('project_type'):
            missing_info.append('project_type')
        if not context.get('data_size'):
            missing_info.append('data_size')
        if not context.get('data_type'):
            missing_info.append('data_type')
        if context.get('project_type') == 'classification' and not context.get('class_count'):
            missing_info.append('class_count')
        
        # Analyze user's message sentiment and content
        user_msg_lower = user_message.lower()
        
        # Generate paragraph-style responses
        if any(word in user_msg_lower for word in ['merhaba', 'selam', 'hello', 'hi']):
            if not context.get('project_type'):
                response = "Merhaba! Size en uygun makine öğrenmesi algoritmalarını bulmaya yardımcı olmaktan memnuniyet duyarım. Projenizin detaylarını anlayarak size özel öneriler geliştirebilirim.\n\nHangi tür bir problem çözmek istediğinizi paylaşabilir misiniz? Bu şekilde size en uygun algoritmaları önerebilirim."
                suggestions = [
                    "Veri sınıflandırması yapacağım",
                    "Sayısal değer tahmini yapmak istiyorum", 
                    "Veri kümelerini gruplamaya ihtiyacım var"
                ]
            else:
                response = f"Merhaba! {context['project_type']} projesi üzerinde çalıştığınızı görüyorum, bu gerçekten ilginç bir alan. Size en uygun algoritmaları önerebilmek için birkaç detay daha öğrenmem gerekiyor."
                suggestions = self._generate_consultation_suggestions(missing_info, context)
        
        elif not context.get('project_type'):
            response = "Projenizin amacını biraz daha detayına inmek istiyorum. Makine öğrenmesinde farklı problem türleri için farklı yaklaşımlar gerekiyor ve size en uygun çözümü sunabilmek için projenizin hedefini anlamam önemli.\n\nHangi tür bir sonuç elde etmeyi hedefliyorsunuz?"
            suggestions = [
                "Verileri kategorilere ayırma (sınıflandırma)",
                "Sayısal değer tahmin etme (regresyon)",
                "Veri gruplarını keşfetme (kümeleme)"
            ]
        
        elif not context.get('data_size'):
            response = f"{context['project_type'].title()} projesi harika bir seçim! Bu alandaki deneyimime dayanarak size çok etkili algoritmalar önerebilirim. Ancak veri setinizin boyutu algoritma seçiminde kritik bir faktör.\n\nKaç tane veri kaydınız var? Bu bilgi sayesinde performans ve hız açısından en uygun algoritmaları seçebilirim."
            suggestions = [
                "1000'den az kayıt (küçük veri)",
                "1000-10000 arası (orta boyut)",
                "10000'den fazla (büyük veri)"
            ]
        
        elif not context.get('data_type'):
            response = "Veri boyutunu öğrendiğim için teşekkürler! Şimdi veri türünü anlamam gerekiyor çünkü farklı veri türleri için optimize edilmiş algoritmalar var. Bu bilgi ile size en uygun ve verimli çözümü önerebilirim.\n\nVerileriniz hangi türde? Bu detay algoritma performansını doğrudan etkiliyor."
            suggestions = [
                "Sayısal veriler (rakamlar, ölçümler)",
                "Kategorik veriler (gruplar, etiketler)",
                "Metin verileri (yazılar, yorumlar)",
                "Görüntü verileri (fotoğraflar, resimler)"
            ]
        
        elif context.get('project_type') == 'classification' and not context.get('class_count'):
            response = "Sınıflandırma projesi için son bir önemli detay kaldı! Kaç farklı kategori veya sınıfınız olduğu algoritma seçimini etkileyecek. İkili sınıflandırma ile çok sınıflı problemler farklı yaklaşımlar gerektiriyor.\n\nVerilerinizi kaç kategoriye ayırmayı planlıyorsunuz?"
            suggestions = [
                "2 kategori (ikili sınıflandırma)",
                "3-10 kategori arası (çoklu sınıf)",
                "10'dan fazla kategori (karmaşık sınıflandırma)"
            ]
        
        else:
            # We have enough info, this shouldn't happen
            response = "Harika! Proje detaylarınızı topladım ve size özel algoritma önerilerini hazırlıyorum. Bir an içinde en uygun seçenekleri sunacağım."
            suggestions = ["Algoritma önerilerini göster"]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
    
    def _generate_consultation_suggestions(self, missing_info: List[str], context: Dict) -> List[str]:
        """
        Generate contextual suggestions based on missing information and context
        """
        if 'proje türü' in missing_info or 'project_type' in missing_info:
            return [
                "Sınıflandırma projesi yapıyorum",
                "Regresyon analizi yapmak istiyorum",
                "Veri kümeleme yapacağım"
            ]
        elif 'veri boyutu' in missing_info or 'data_size' in missing_info:
            return [
                "Küçük veri setim var (1000<)",
                "Orta boyut veri (1000-10000)",
                "Büyük veri setim var (10000+)"
            ]
        elif 'veri türü' in missing_info or 'data_type' in missing_info:
            return [
                "Sayısal verilerle çalışıyorum",
                "Kategorik verilerim var",
                "Metin verileri işliyorum"
            ]
        else:
            return [
                "Algoritma önerilerini ver",
                "Performans karşılaştırması yap",
                "Hangi metrik kullanmalıyım?"
            ]
    
    def _ai_consultation_response(self, user_message: str, context: Dict) -> Dict:
        """
        Fallback to advanced AI when OpenAI fails
        """
        return self._advanced_ai_consultation_response(user_message, context)
    
    def _template_consultation_response(self, context: Dict) -> Dict:
        """
        Template-based consultation when AI is not available
        """
        if not context.get('project_type'):
            return {
                "response": "Merhaba! Projeniz için en uygun algoritmaları önerebilmek için biraz daha bilgiye ihtiyacım var. Hangi tür bir makine öğrenmesi problemi çözmek istiyorsunuz?",
                "suggestions": [
                    "Veri sınıflandırması yapacağım",
                    "Sayısal değer tahmini (regresyon)",
                    "Veri kümeleme işlemi"
                ],
                "success": True
            }
        elif not context.get('data_size'):
            return {
                "response": f"Harika! {context['project_type']} projesi için size yardımcı olabilirim. Veri setinizin boyutu nasıl?",
                "suggestions": [
                    "1000'den az veri",
                    "1000-10000 arası veri",
                    "10000+ büyük veri seti"
                ],
                "success": True
            }
        else:
            return self._get_emergency_fallback()
    
    def _template_recommendations(self, context: Dict, recommendations: List[Dict]) -> Dict:
        """
        Template-based recommendations when AI is not available
        """
        # Track recommended algorithms
        for rec in recommendations:
            algo_name = rec.get('algorithm', '').lower()
            if algo_name and algo_name not in self.conversation_context['discussed_algorithms']:
                self.conversation_context['discussed_algorithms'].append(algo_name)
        
        response = f"🎯 **{context.get('project_type', 'ML').title()} Projesi için Önerilerim:**\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            response += f"**{i}. {rec['algorithm']}**\n"
            response += f"   • Güven Skoru: {rec['confidence_score']:.2f}\n"
            response += f"   • {rec.get('description', 'Güvenilir algoritma')}\n\n"
        
        response += "Bu algoritmaların hangisi hakkında daha fazla bilgi almak istersiniz?"
        
        suggestions = [rec['algorithm'] for rec in recommendations[:3]]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True
        }
    
    def _generate_smart_suggestions(self, context: Dict, recommendations: List[Dict]) -> List[str]:
        """
        Generate contextual suggestions based on recommendations
        """
        suggestions = []
        
        if recommendations:
            suggestions.append(f"{recommendations[0]['algorithm']} hakkında detay")
            suggestions.append("Implementasyon örneği")
            suggestions.append("Performans karşılaştırması")
        
        return suggestions[:3]
    
    def _generate_context_suggestions(self, missing_info: List[str]) -> List[str]:
        """
        Generate suggestions based on missing information
        """
        if 'proje türü' in str(missing_info):
            return [
                "Sınıflandırma projesi yapıyorum",
                "Regresyon analizi yapacağım",
                "Veri kümeleme yapacağım"
            ]
        elif 'veri boyutu' in str(missing_info):
            return [
                "Küçük veri setim var (1000<)",
                "Orta boyut veri (1000-10000)",
                "Büyük veri setim var (10000+)"
            ]
        else:
            return [
                "Daha fazla detay ver",
                "Örnek göster",
                "Başka yaklaşım"
            ]
    
    def _get_enhanced_explanation(self, algorithm: str, context: Dict) -> str:
        """
        Get enhanced paragraph-style explanation for each algorithm
        """
        project_type = context.get('project_type', 'general')
        data_size = context.get('data_size', 'medium')
        
        explanations = {
            'XGBoost': {
                'classification': f"Bu gradient boosting algoritması, sınıflandırma problemlerinde çok yüksek doğruluk oranları sağlar. Özellikle {data_size} boyuttaki veri setlerde mükemmel sonuçlar verir çünkü birçok zayıf öğreniciyi birleştirerek güçlü bir model oluşturur. Eksik verilerle bile başarılı çalışması ve özellik önemini göstermesi büyük avantajları.",
                'regression': f"Sayısal tahminlerde üstün performans gösteren bu algoritma, karmaşık veri ilişkilerini yakalama konusunda uzman. {data_size.title()} veri setinizde trend analizi ve pattern recognition konularında çok başarılı olacak.",
                'general': "Hemen hemen her machine learning probleminde güvenle kullanabileceğiniz, endüstri standardı bir algoritma. Kaggle yarışmalarının favorisi olmasının sebebi yüksek performansı ve esnekliği."
            },
            'Random Forest': {
                'classification': f"Karar ağaçlarının kollektif gücünü kullanarak overfitting problemini çözen akıllı bir yaklaşım. {data_size.title()} veri setinizde hem hızlı çalışacak hem de yorumlanabilir sonuçlar verecek. Özellik önemini görmek için ideal.",
                'regression': f"Tahmin problemlerinde güvenilirlik arıyorsanız mükemmel bir seçim. Birçok karar ağacının oybirliği ile tahmin yaptığı için tek bir ağaca göre çok daha stabil sonuçlar verir.",
                'general': "Başlangıç için ideal çünkü hiperparametre ayarlamaya çok ihtiyaç duymaz ve neredeyse her durumda makul sonuçlar verir. Güvenilir bir algoritma."
            },
            'Logistic Regression': {
                'classification': f"Basitliği ve etkinliği ile öne çıkan bu algoritma, {data_size} veri setlerde hızlı sonuçlar verir. İkili sınıflandırmada özellikle başarılı ve sonuçları anlamak çok kolay. Doğrusal ilişkileri çok iyi yakalar.",
                'general': "Machine learning'e yeni başlayanlar için mükemmel bir başlangıç noktası. Hem hızlı hem de yorumlanabilir sonuçlar verir."
            },
            'SVM': {
                'classification': f"Karmaşık sınır çizgilerini çizme konusunda uzman bu algoritma, özellikle doğrusal olmayan ilişkilerin olduğu durumlarda çok başarılı. {data_size} veri setlerde kernel trick sayesinde yüksek boyutlu problemleri çözebilir.",
                'general': "Güçlü matematik temeli olan, teorik olarak sağlam bir algoritma. Özellikle yüksek boyutlu verilerde etkili."
            },
            'Neural Network': {
                'classification': f"İnsan beyninden ilham alan bu algoritma, çok karmaşık pattern'leri öğrenebilir. {data_size} veri setiniz büyükse harika sonuçlar verecek, ancak parametre ayarlaması biraz sabır gerektirir.",
                'general': "Derin öğrenmenin kapısını açan temel algoritma. Karmaşık problemlerde çok güçlü ama yeterli veri gerektir."
            }
        }
        
        if algorithm in explanations:
            if project_type in explanations[algorithm]:
                return explanations[algorithm][project_type]
            else:
                return explanations[algorithm]['general']
        else:
            return f"Bu algoritma {project_type} problemlerde güvenilir sonuçlar verir ve veri setinizin karakteristikleriyle uyumlu çalışacaktır."
    
    def _get_emergency_fallback(self) -> Dict:
        """
        Emergency response when everything fails - should be used sparingly
        """
        # Generate diverse fallback responses with better context awareness
        fallback_responses = [
            "🤔 **Özür dilerim, sorunuzu tam anlayamadım.** Makine öğrenmesi projeniz hakkında daha net bilgi verebilir misiniz? Hangi tür bir problem çözmeye çalışıyorsunuz?",
            
            "🔍 **Biraz daha detay verebilir misiniz?** Projenizin amacını ve hangi tür verilerle çalıştığınızı anlamak istiyorum. Bu şekilde size daha iyi yardım edebilirim.",
            
            "💡 **Size daha iyi yardım edebilmek için** projenizin detaylarını öğrenmek istiyorum. Hangi alanda çalışıyorsunuz ve ne tür bir analiz yapmak istiyorsunuz?",
            
            "🎯 **Anladığım kadarıyla** bir makine öğrenmesi projesi üzerinde çalışıyorsunuz. Hangi tür bir problem çözmeye odaklanıyorsunuz? Sınıflandırma, tahmin, yoksa başka bir şey mi?"
        ]
        
        # Use hash of current time to ensure variety but avoid complete randomness
        import time
        response_index = int(time.time()) % len(fallback_responses)
        
        return {
            "response": fallback_responses[response_index],
            "suggestions": [
                "Veri sınıflandırması yapacağım",
                "Sayısal tahmin modeli geliştiriyorum", 
                "Veri kümeleme işlemi yapacağım",
                "Hangi algoritma en uygun?"
            ],
            "success": True
        } 