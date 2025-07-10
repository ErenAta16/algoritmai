"""
Enhanced OpenAI Service with Session Management and Context-Aware Fallback
Fixes all major issues: context persistence, response diversity, intelligent fallback
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from services.algorithm_recommender import AlgorithmRecommender
from services.session_manager import session_manager
from services.context_aware_fallback import context_aware_fallback

load_dotenv()
logger = logging.getLogger(__name__)

class EnhancedOpenAIService:
    """
    Enhanced OpenAI Service with session management and intelligent fallback
    """
    
    def __init__(self):
        """Initialize enhanced OpenAI service"""
        self.algorithm_recommender = AlgorithmRecommender()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize OpenAI client
        self._setup_openai_client()
        
        # System prompts
        self.system_prompt = """Sen "AlgoMentor" adında deneyimli bir makine öğrenmesi uzmanısın. 
        Kullanıcılarla samimi ve yardımsever şekilde konuşuyorsun. 
        
        Özellikler:
        - Doğal, akıcı paragraflar halinde konuş
        - Kişisel deneyimlerini paylaş
        - Yaratıcı çözümler öner
        - Teknik bilgiyi basit örneklerle anlat
        - Her zaman Türkçe yanıt ver
        
        Robotik cevaplar verme, gerçek bir mentor gibi davran!"""
        
        logger.info("✅ Enhanced OpenAI Service initialized")
    
    def _setup_openai_client(self):
        """Setup OpenAI client with proper error handling"""
        api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key and api_key != 'your_openai_api_key_here':
            if not api_key.startswith('sk-') or len(api_key) < 20:
                logger.error("❌ Invalid OpenAI API key format")
                self.openai_enabled = False
                self.openai_client = None
            else:
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
            logger.warning("⚠️ OpenAI API key not found, using intelligent fallback")
    
    async def get_chat_response_async(self, user_message: str, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict:
        """Async version of get_chat_response"""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            self.get_chat_response,
            user_message,
            conversation_history,
            session_id
        )
        
        return result
    
    def get_chat_response(self, user_message: str, conversation_history: Optional[List[Dict]] = None, session_id: str = None) -> Dict:
        """
        Enhanced chat response with session management and context persistence
        """
        try:
            # Handle None or empty messages
            if not user_message or not user_message.strip():
                return self._get_welcome_message()
            
            user_message = user_message.strip()
            
            # Get or create session
            if not session_id:
                session_id = session_manager.get_or_create_session()
            else:
                session_id = session_manager.get_or_create_session(session_id)
            
            logger.info(f"🔍 Processing message: '{user_message}' [Session: {session_id}]")
            
            # Merge conversation history with session context
            merged_history = session_manager.merge_conversation_history(session_id, conversation_history or [])
            
            # Extract persistent context
            context = session_manager.extract_persistent_context(session_id, user_message, merged_history)
            
            # Get response diversity context
            diversity_context = session_manager.get_response_diversity_context(session_id, user_message)
            
            # Add diversity information to context
            context['diversity_context'] = diversity_context
            
            logger.info(f"📊 Context: {context}")
            
            # Determine response type
            response_type = self._determine_response_type(user_message, context)
            logger.info(f"🎯 Response Type: {response_type}")
            
            # Generate response
            response = self._generate_intelligent_response(user_message, context, response_type)
            
            # Track algorithm mentions
            recommendations = response.get('recommendations', [])
            session_manager.track_algorithm_mentions(session_id, user_message, recommendations)
            
            # Store response for diversity tracking
            session_manager.store_response_for_diversity(session_id, user_message, response['response'])
            
            # Update session context
            session_manager.update_session_context(session_id, {
                'conversation_context': context.get('conversation_context', {}),
                'project_context': context
            })
            
            # Add session ID to response
            response['session_id'] = session_id
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in get_chat_response: {str(e)}")
            return self._get_emergency_fallback()
    
    def _determine_response_type(self, user_message: str, context: Dict) -> str:
        """Determine appropriate response type based on context"""
        text_lower = user_message.lower()
        
        # Check for recommendation responses
        if self._is_responding_to_recommendations(user_message, context):
            return 'recommendation_response'
        
        # Check for algorithm-specific questions
        if self._is_algorithm_question(user_message, context):
            return 'algorithm_question'
        
        # Check for code requests
        if any(word in text_lower for word in ['kod', 'örnek', 'implement', 'python', 'nasıl yapılır']):
            return 'code_request'
        
        # Check for comparison requests
        if any(word in text_lower for word in ['karşılaştır', 'hangisi', 'fark', 'vs', 'compare']):
            return 'comparison_request'
        
        # Check for alternative requests
        if any(word in text_lower for word in ['başka', 'alternatif', 'farklı', 'diğer']) and \
           any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye']):
            return 'alternative_request'
        
        # Check if ready for recommendations - more aggressive
        if self._is_ready_for_recommendations(context):
            return 'recommendation_ready'
        
        # Check for algorithm requests
        if any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye', 'recommend']):
            return 'algorithm_request'
        
        # Check for project type mentions
        if any(word in text_lower for word in ['sınıflandırma', 'siniflandirma', 'classification', 'regresyon', 'regression', 'kümeleme', 'kumeleme', 'clustering']):
            return 'project_info_collection'
        
        # Check for data size mentions
        if any(word in text_lower for word in ['küçük', 'kucuk', 'small', 'büyük', 'buyuk', 'large', 'orta', 'medium', 'veri', 'data']):
            return 'project_info_collection'
        
        # Default to consultation
        return 'consultation'
    
    def _generate_intelligent_response(self, user_message: str, context: Dict, response_type: str) -> Dict:
        """Generate intelligent response with OpenAI or fallback"""
        # Try OpenAI first if available
        if self.openai_enabled and self.openai_client:
            try:
                return self._generate_openai_response(user_message, context, response_type)
            except Exception as e:
                logger.warning(f"⚠️ OpenAI failed, using fallback: {str(e)[:100]}...")
        
        # Use context-aware fallback
        return self._generate_fallback_response(user_message, context, response_type)
    
    def _generate_openai_response(self, user_message: str, context: Dict, response_type: str) -> Dict:
        """Generate response using OpenAI"""
        # Build context-aware prompt
        context_prompt = self._build_context_prompt(context, response_type)
        
        # Add diversity instruction if needed
        diversity_context = context.get('diversity_context', {})
        if diversity_context.get('similar_responses'):
            context_prompt += "\n\nÖNEMLİ: Bu soruya daha önce benzer cevaplar verdin. Şimdi farklı bir yaklaşım, farklı örnekler ve farklı ifadeler kullan. Önceki cevaplarını tekrar etme."
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{context_prompt}\n\nKullanıcı mesajı: {user_message}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            timeout=30
        )
        
        ai_response = response.choices[0].message.content
        
        # Generate suggestions
        suggestions = self._generate_contextual_suggestions(context, response_type)
        
        return {
            "response": ai_response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": True,
            "context_aware": True
        }
    
    def _generate_fallback_response(self, user_message: str, context: Dict, response_type: str) -> Dict:
        """Generate response using context-aware fallback"""
        # Handle specific response types
        if response_type == 'recommendation_ready':
            return self._generate_algorithm_recommendations(user_message, context)
        elif response_type == 'project_info_collection':
            return self._handle_project_info_collection(user_message, context)
        elif response_type == 'algorithm_request':
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        elif response_type == 'alternative_request':
            return self._handle_alternative_request(user_message, context)
        elif response_type == 'recommendation_response':
            return self._handle_recommendation_response(user_message, context)
        elif response_type == 'algorithm_question':
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        elif response_type == 'code_request':
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        elif response_type == 'comparison_request':
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        else:
            return context_aware_fallback.generate_context_aware_response(user_message, context)
    
    def _generate_algorithm_recommendations(self, user_message: str, context: Dict) -> Dict:
        """Generate algorithm recommendations with context awareness"""
        try:
            # Check if we have enough context for meaningful recommendations
            project_type = context.get('project_type')
            data_size = context.get('data_size')
            
            # If no context, ask for more information
            if not project_type or not data_size:
                response = "🤔 **Daha iyi öneriler verebilmek için biraz daha bilgiye ihtiyacım var:**\n\n"
                response += "**Projenizin türü nedir?**\n"
                response += "• Sınıflandırma (Classification)\n"
                response += "• Regresyon (Regression)\n"
                response += "• Kümeleme (Clustering)\n"
                response += "• Zaman serisi analizi\n\n"
                response += "**Veri boyutunuz nasıl?**\n"
                response += "• Küçük (< 1000 örnek)\n"
                response += "• Orta (1000-10000 örnek)\n"
                response += "• Büyük (> 10000 örnek)"
                
                suggestions = [
                    "Sınıflandırma projesi yapıyorum",
                    "Regresyon projesi yapıyorum", 
                    "Kümeleme projesi yapıyorum",
                    "Zaman serisi analizi yapıyorum"
                ]
                
                return {
                    "response": response,
                    "suggestions": suggestions,
                    "success": True,
                    "ai_powered": False,
                    "context_aware": True,
                    "needs_more_context": True
                }
            
            # Get recommendations from algorithm recommender
            recommendations = self.algorithm_recommender.get_recommendations(
                project_type=project_type,
                data_size=data_size,
                data_type=context.get('data_type'),
                complexity_preference=context.get('complexity_preference'),
                top_n=3
            )
            
            if not recommendations:
                return context_aware_fallback.generate_context_aware_response(user_message, context)
            
            # Build response
            response = f"🎯 **{project_type.title()} Projesi için Önerilerim:**\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                response += f"**{i}. {rec['algorithm']}**\n"
                response += f"   • Güven Skoru: {rec['confidence_score']:.1f}/5.0\n"
                response += f"   • Karmaşıklık: {rec['complexity']}\n"
                response += f"   • {rec.get('description', 'Güvenilir algoritma seçimi')}\n\n"
            
            response += "Bu algoritmalardan hangisi hakkında daha fazla bilgi almak istersiniz?"
            
            suggestions = [f"{rec['algorithm']} hakkında bilgi ver" for rec in recommendations]
            suggestions.append("Kod örneği göster")
            suggestions.append("Performans karşılaştırması")
            
            return {
                "response": response,
                "suggestions": suggestions,
                "recommendations": recommendations,
                "success": True,
                "ai_powered": False,
                "context_aware": True
            }
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {str(e)}")
            return context_aware_fallback.generate_context_aware_response(user_message, context)
    
    def _handle_alternative_request(self, user_message: str, context: Dict) -> Dict:
        """Handle alternative algorithm requests"""
        conversation_context = context.get('conversation_context', {})
        discussed_algorithms = conversation_context.get('discussed_algorithms', [])
        
        # Get alternative recommendations
        try:
            all_recommendations = self.algorithm_recommender.get_recommendations(
                project_type=context.get('project_type'),
                data_size=context.get('data_size'),
                data_type=context.get('data_type'),
                top_n=6
            )
            
            # Filter out discussed algorithms
            alternatives = [rec for rec in all_recommendations 
                          if rec['algorithm'].lower() not in [algo.lower() for algo in discussed_algorithms]]
            
            if not alternatives:
                alternatives = all_recommendations  # If no alternatives, show all
            
            response = "🔄 **Alternatif Algoritma Önerileri:**\n\n"
            
            for i, rec in enumerate(alternatives[:3], 1):
                response += f"**{i}. {rec['algorithm']}**\n"
                response += f"   • Güven Skoru: {rec['confidence_score']:.1f}/5.0\n"
                response += f"   • Karmaşıklık: {rec['complexity']}\n"
                response += f"   • {rec.get('description', 'Alternatif çözüm')}\n\n"
            
            response += "Bu alternatiflerden hangisi ilginizi çekiyor?"
            
            suggestions = [f"{rec['algorithm']} tercih ediyorum" for rec in alternatives[:3]]
            suggestions.append("Daha fazla alternatif göster")
            
            return {
                "response": response,
                "suggestions": suggestions,
                "recommendations": alternatives,
                "success": True,
                "ai_powered": False,
                "context_aware": True
            }
            
        except Exception as e:
            logger.error(f"❌ Alternative request error: {str(e)}")
            return context_aware_fallback.generate_context_aware_response(user_message, context)
    
    def _handle_recommendation_response(self, user_message: str, context: Dict) -> Dict:
        """Handle user responses to recommendations"""
        text_lower = user_message.lower()
        conversation_context = context.get('conversation_context', {})
        last_recommendations = conversation_context.get('last_recommendations', [])
        
        # Check response type
        if any(word in text_lower for word in ['hayır', 'farklı', 'başka', 'alternatif']):
            return self._handle_alternative_request(user_message, context)
        elif any(word in text_lower for word in ['neden', 'avantaj', 'dezavantaj', 'açıkla']):
            return self._explain_recommendations(user_message, context, last_recommendations)
        elif any(word in text_lower for word in ['kod', 'örnek', 'implement']):
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        else:
            # General positive response
            response = "🎉 **Harika! Seçiminizi beğendiğinize sevindim.**\n\n"
            response += "Size nasıl yardımcı olabilirim?\n"
            response += "• Kod örneği gösterebilirim\n"
            response += "• Algoritmanın avantajlarını açıklayabilirim\n"
            response += "• Implementasyon ipuçları verebilirim\n"
            response += "• Başka algoritmalar önerebilirim"
            
            suggestions = [
                "Kod örneği göster",
                "Avantajlarını açıkla",
                "Implementasyon ipuçları",
                "Başka algoritma öner"
            ]
            
            return {
                "response": response,
                "suggestions": suggestions,
                "success": True,
                "ai_powered": False,
                "context_aware": True
            }
    
    def _handle_project_info_collection(self, user_message: str, context: Dict) -> Dict:
        """Handle project information collection"""
        text_lower = user_message.lower()
        
        # Check what information we have and what we need
        project_type = context.get('project_type')
        data_size = context.get('data_size')
        data_type = context.get('data_type')
        
        response = "📋 **Proje Bilgilerinizi Topluyorum:**\n\n"
        
        # Update context based on current message
        if any(word in text_lower for word in ['sınıflandırma', 'siniflandirma', 'classification', 'classify']):
            project_type = 'classification'
            response += "✅ **Proje Türü:** Sınıflandırma\n"
        elif any(word in text_lower for word in ['regresyon', 'regression']):
            project_type = 'regression'
            response += "✅ **Proje Türü:** Regresyon\n"
        elif any(word in text_lower for word in ['kümeleme', 'kumeleme', 'clustering', 'cluster']):
            project_type = 'clustering'
            response += "✅ **Proje Türü:** Kümeleme\n"
        
        if any(word in text_lower for word in ['küçük', 'kucuk', 'small', 'az']):
            data_size = 'small'
            response += "✅ **Veri Boyutu:** Küçük (< 1000 örnek)\n"
        elif any(word in text_lower for word in ['orta', 'medium']):
            data_size = 'medium'
            response += "✅ **Veri Boyutu:** Orta (1000-10000 örnek)\n"
        elif any(word in text_lower for word in ['büyük', 'buyuk', 'large', 'çok']):
            data_size = 'large'
            response += "✅ **Veri Boyutu:** Büyük (> 10000 örnek)\n"
        
        # Check if we have enough info for recommendations
        if project_type and data_size:
            response += "\n🎯 **Artık size özel algoritma önerileri verebilirim!**\n\n"
            response += "Hangi algoritmaları önermemi istersiniz?"
            
            suggestions = [
                "Algoritma önerileri göster",
                "En iyi algoritmaları listele",
                "Performans karşılaştırması yap"
            ]
        else:
            response += "\n❓ **Hala ihtiyacım olan bilgiler:**\n"
            
            if not project_type:
                response += "• Projenizin türü (sınıflandırma/regresyon/kümeleme)\n"
            if not data_size:
                response += "• Veri boyutunuz (küçük/orta/büyük)\n"
            
            suggestions = [
                "Sınıflandırma projesi yapıyorum",
                "Regresyon projesi yapıyorum",
                "Kümeleme projesi yapıyorum",
                "Veri boyutum küçük",
                "Veri boyutum orta",
                "Veri boyutum büyük"
            ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True,
            "project_type": project_type,
            "data_size": data_size
        }
    
    def _explain_recommendations(self, user_message: str, context: Dict, recommendations: List[Dict]) -> Dict:
        """Explain why recommendations were made"""
        if not recommendations:
            return context_aware_fallback.generate_context_aware_response(user_message, context)
        
        response = "🔍 **Neden Bu Algoritmaları Önerdim:**\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            response += f"**{i}. {rec['algorithm']}**\n"
            
            # Add explanation based on match reasons
            reasons = rec.get('match_reasons', [])
            if reasons:
                response += "   Çünkü:\n"
                for reason in reasons:
                    response += f"   • {reason}\n"
            else:
                response += f"   • Projenizin türü ({context.get('project_type', 'genel')}) için uygun\n"
                response += f"   • Veri boyutunuz ({context.get('data_size', 'orta')}) ile uyumlu\n"
                response += f"   • Güven skoru yüksek ({rec['confidence_score']:.1f}/5.0)\n"
            
            response += "\n"
        
        response += "Hangi algoritma hakkında daha detaylı bilgi almak istersiniz?"
        
        suggestions = [f"{rec['algorithm']} detayları" for rec in recommendations[:3]]
        suggestions.append("Kod örnekleri göster")
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _is_responding_to_recommendations(self, user_message: str, context: Dict) -> bool:
        """Check if user is responding to previous recommendations"""
        text_lower = user_message.lower()
        
        # Check if there were recent recommendations
        conversation_context = context.get('conversation_context', {})
        last_recommendations = conversation_context.get('last_recommendations', [])
        
        if not last_recommendations:
            return False
        
        # Response indicators
        response_indicators = [
            'hayır', 'evet', 'daha iyi', 'tercih', 'seçmek', 'istiyorum',
            'farklı', 'başka', 'alternatif', 'neden', 'avantaj', 'dezavantaj',
            'kod', 'örnek', 'implement'
        ]
        
        return any(indicator in text_lower for indicator in response_indicators)
    
    def _is_algorithm_question(self, user_message: str, context: Dict) -> bool:
        """Check if user is asking about specific algorithms"""
        text_lower = user_message.lower()
        
        # Algorithm names
        algorithms = [
            'xgboost', 'random forest', 'svm', 'neural', 'logistic',
            'k-means', 'kmeans', 'dbscan', 'naive bayes', 'decision tree'
        ]
        
        # Question indicators
        question_indicators = [
            'nedir', 'nasıl çalışır', 'açıkla', 'anlat', 'avantaj', 'dezavantaj',
            'ne zaman kullan', 'hangi durumda', 'bilgi ver', 'hakkında'
        ]
        
        has_algorithm = any(algo in text_lower for algo in algorithms)
        has_question = any(indicator in text_lower for indicator in question_indicators)
        
        return has_algorithm and has_question
    
    def _is_ready_for_recommendations(self, context: Dict) -> bool:
        """Check if we have enough context for recommendations"""
        required_fields = ['project_type', 'data_size', 'data_type']
        return all(context.get(field) for field in required_fields)
    
    def _build_context_prompt(self, context: Dict, response_type: str) -> str:
        """Build context-aware prompt for OpenAI"""
        prompt = f"Kullanıcı profili ve proje bilgileri:\n"
        
        if context.get('project_type'):
            prompt += f"- Proje türü: {context['project_type']}\n"
        if context.get('data_size'):
            prompt += f"- Veri boyutu: {context['data_size']}\n"
        if context.get('data_type'):
            prompt += f"- Veri türü: {context['data_type']}\n"
        
        conversation_context = context.get('conversation_context', {})
        if conversation_context.get('discussed_algorithms'):
            prompt += f"- Konuşulan algoritmalar: {', '.join(conversation_context['discussed_algorithms'])}\n"
        
        prompt += f"\nYanıt türü: {response_type}\n"
        
        return prompt
    
    def _generate_contextual_suggestions(self, context: Dict, response_type: str) -> List[str]:
        """Generate contextual suggestions based on context"""
        suggestions = []
        
        if response_type == 'consultation':
            if not context.get('project_type'):
                suggestions = [
                    "Sınıflandırma projesi yapıyorum",
                    "Regresyon analizi yapmak istiyorum",
                    "Veri kümeleme yapacağım"
                ]
            elif not context.get('data_size'):
                suggestions = [
                    "Küçük veri setim var",
                    "Orta boyutta veri setim var",
                    "Büyük veri setim var"
                ]
            else:
                suggestions = [
                    "Algoritma önerisi istiyorum",
                    "Kod örneği göster",
                    "Performans karşılaştırması"
                ]
        else:
            suggestions = [
                "Kod örneği göster",
                "Başka algoritma öner",
                "Performans karşılaştırması",
                "Detaylı açıklama"
            ]
        
        return suggestions
    
    def _get_welcome_message(self) -> Dict:
        """Get welcome message for new conversations"""
        return {
            "response": "Merhaba! Ben AlgoMentor, sizin kişisel makine öğrenmesi danışmanınızım. 🎯\n\nSize en uygun algoritmaları bulmak için projenizin detaylarını öğrenmek istiyorum.\n\nHangi tür bir makine öğrenmesi projesi üzerinde çalışıyorsunuz?",
            "suggestions": [
                "Sınıflandırma projesi yapıyorum",
                "Regresyon analizi yapmak istiyorum",
                "Veri kümeleme işlemi yapacağım",
                "Anomali tespiti yapmak istiyorum"
            ],
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _get_emergency_fallback(self) -> Dict:
        """Emergency fallback when everything fails"""
        return {
            "response": "Üzgünüm, şu anda bir teknik sorun yaşıyorum. Lütfen mesajınızı tekrar gönderir misiniz?",
            "suggestions": [
                "Tekrar dene",
                "Algoritma önerisi istiyorum",
                "Yardım",
                "Baştan başla"
            ],
            "success": False,
            "ai_powered": False,
            "context_aware": False
        }

# Global enhanced service instance
enhanced_openai_service = EnhancedOpenAIService() 