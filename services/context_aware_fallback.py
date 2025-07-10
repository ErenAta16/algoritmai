"""
Context-Aware Fallback System for AI Algorithm Consultant
Provides intelligent responses when OpenAI API is unavailable
"""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ContextAwareFallback:
    """
    Advanced fallback system that provides context-aware responses
    when OpenAI API is unavailable
    """
    
    def __init__(self):
        self.response_templates = {
            'algorithm_request': {
                'classification': [
                    "Sınıflandırma projesi için harika seçenekleriniz var! Verilerinizin boyutuna göre şu algoritmaları önerebilirim:",
                    "Sınıflandırma konusunda size yardımcı olabilirim. Projenizin özelliklerine göre en uygun algoritmaları bulalım:",
                    "Mükemmel! Sınıflandırma algoritmaları konusunda deneyimim var. Size özel öneriler hazırlayabilirim:"
                ],
                'regression': [
                    "Regresyon analizi için çok güzel seçenekler var! Verilerinizin yapısına göre öneriler sunabilirim:",
                    "Regresyon projesi ilginç! Hangi değerleri tahmin etmeye çalışıyorsunuz? Buna göre algoritma önerebilirim:",
                    "Regresyon konusunda size yardımcı olabilirim. Projenizin detaylarına göre en uygun çözümleri bulalım:"
                ],
                'clustering': [
                    "Kümeleme analizi gerçekten faydalı! Verilerinizi nasıl gruplamak istediğinize göre öneriler verebilirim:",
                    "Veri kümeleme konusunda size yardımcı olabilirim. Hangi tür gruplama yapmak istiyorsunuz?",
                    "Kümeleme algoritmaları konusunda deneyimim var. Projenizin amacına göre en uygun yöntemi bulalım:"
                ]
            },
            'alternative_requests': [
                "Tabii ki! Size farklı algoritma alternatifleri sunabilirim. Hangi kriterlere odaklanmak istersiniz?",
                "Elbette! Başka seçenekler de var. Projenizin özelliklerine göre alternatif algoritmaları değerlendirelim:",
                "Kesinlikle! Farklı yaklaşımlar deneyebiliriz. Hangi konularda daha fazla bilgi almak istersiniz?"
            ],
            'algorithm_explanation': {
                'xgboost': {
                    'intro': "XGBoost gerçekten güçlü bir algoritma! Gradient boosting ailesinden geliyor.",
                    'strengths': ["Yüksek doğruluk oranları", "Overfitting'e karşı dayanıklı", "Özellik önemini gösterir"],
                    'weaknesses': ["Hiperparametre ayarlaması gerekebilir", "Küçük veri setlerinde overkill olabilir"],
                    'use_cases': ["Sınıflandırma", "Regresyon", "Ranking problemleri"]
                },
                'random_forest': {
                    'intro': "Random Forest çok güvenilir bir algoritma! Birçok karar ağacını birleştiriyor.",
                    'strengths': ["Overfitting riski düşük", "Yorumlanabilir", "Hiperparametre ayarı minimal"],
                    'weaknesses': ["Çok büyük veri setlerinde yavaş", "Bellek kullanımı yüksek"],
                    'use_cases': ["Sınıflandırma", "Regresyon", "Özellik seçimi"]
                },
                'k-means': {
                    'intro': "K-means clustering için mükemmel bir seçim! Basit ama etkili.",
                    'strengths': ["Anlaşılır algoritma", "Hızlı çalışır", "Büyük verilerle başa çıkar"],
                    'weaknesses': ["K değerini önceden belirlemek gerekir", "Outlier'lara hassas"],
                    'use_cases': ["Müşteri segmentasyonu", "Pazar analizi", "Veri gruplama"]
                }
            },
            'consultation_questions': {
                'project_type_unknown': [
                    "Projenizin amacını daha iyi anlayabilmek için: Hangi tür bir analiz yapmak istiyorsunuz?",
                    "Size daha iyi yardımcı olabilmek için projenizin hedefini öğrenmek istiyorum. Ne yapmaya çalışıyorsunuz?",
                    "Hangi konuda çalışıyorsunuz? Sınıflandırma, regresyon, kümeleme gibi hangi alanda yardıma ihtiyacınız var?"
                ],
                'data_size_unknown': [
                    "Verilerinizin boyutu nasıl? Küçük bir veri seti mi yoksa büyük veri ile mi çalışıyorsunuz?",
                    "Kaç tane veri noktanız var? Bu algoritma seçiminde önemli bir faktör.",
                    "Veri setinizin büyüklüğü algoritma seçimini etkiler. Hangi boyutta verilerle çalışıyorsunuz?"
                ],
                'data_type_unknown': [
                    "Verilerinizin türü nasıl? Sayısal, kategorik, metin ya da görüntü verileri mi?",
                    "Hangi tür verilerle çalışıyorsunuz? Bu, algoritma seçiminde kritik bir faktör.",
                    "Verilerinizin yapısını öğrenebilir miyim? Sayısal değerler mi, kategoriler mi, yoksa başka türde mi?"
                ]
            }
        }
        
        self.algorithm_database = {
            'classification': {
                'small_data': ['Logistic Regression', 'Naive Bayes', 'KNN', 'Decision Tree'],
                'medium_data': ['Random Forest', 'SVM', 'Neural Networks', 'XGBoost'],
                'large_data': ['XGBoost', 'LightGBM', 'Neural Networks', 'Ensemble Methods']
            },
            'regression': {
                'small_data': ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN'],
                'medium_data': ['Random Forest', 'SVR', 'Neural Networks', 'XGBoost'],
                'large_data': ['XGBoost', 'LightGBM', 'Neural Networks', 'Ensemble Methods']
            },
            'clustering': {
                'small_data': ['K-Means', 'Hierarchical Clustering', 'DBSCAN'],
                'medium_data': ['K-Means', 'DBSCAN', 'Mean Shift', 'Spectral Clustering'],
                'large_data': ['K-Means', 'Mini-Batch K-Means', 'DBSCAN', 'OPTICS']
            }
        }
    
    def generate_context_aware_response(self, user_message: str, context: Dict) -> Dict:
        """
        Generate intelligent context-aware response based on conversation state
        """
        try:
            # Analyze user intent
            intent = self._analyze_user_intent(user_message, context)
            
            # Generate appropriate response
            if intent == 'algorithm_request':
                return self._handle_algorithm_request(user_message, context)
            elif intent == 'alternative_request':
                return self._handle_alternative_request(user_message, context)
            elif intent == 'algorithm_explanation':
                return self._handle_algorithm_explanation(user_message, context)
            elif intent == 'consultation_needed':
                return self._handle_consultation(user_message, context)
            elif intent == 'comparison_request':
                return self._handle_comparison_request(user_message, context)
            elif intent == 'code_request':
                return self._handle_code_request(user_message, context)
            else:
                return self._handle_general_inquiry(user_message, context)
                
        except Exception as e:
            logger.error(f"❌ Context-aware fallback error: {str(e)}")
            return self._get_safe_fallback(user_message)
    
    def _analyze_user_intent(self, user_message: str, context: Dict) -> str:
        """Analyze user intent based on message and context"""
        text_lower = user_message.lower()
        
        # Check for algorithm requests
        if any(word in text_lower for word in ['algoritma', 'öner', 'öneri', 'tavsiye', 'recommend']):
            return 'algorithm_request'
        
        # Check for alternative requests
        if any(word in text_lower for word in ['başka', 'alternatif', 'farklı', 'diğer']):
            return 'alternative_request'
        
        # Check for algorithm explanations
        if any(word in text_lower for word in ['açıkla', 'anlat', 'nedir', 'nasıl çalışır', 'avantaj', 'dezavantaj']):
            return 'algorithm_explanation'
        
        # Check for code requests
        if any(word in text_lower for word in ['kod', 'örnek', 'implement', 'python', 'nasıl yapılır']):
            return 'code_request'
        
        # Check for comparison requests
        if any(word in text_lower for word in ['karşılaştır', 'hangisi', 'fark', 'vs', 'compare']):
            return 'comparison_request'
        
        # Check if consultation is needed
        project_type = context.get('project_type')
        data_size = context.get('data_size')
        data_type = context.get('data_type')
        
        if not project_type or not data_size or not data_type:
            return 'consultation_needed'
        
        return 'general_inquiry'
    
    def _handle_algorithm_request(self, user_message: str, context: Dict) -> Dict:
        """Handle algorithm recommendation requests"""
        project_type = context.get('project_type', 'classification')
        data_size = context.get('data_size', 'medium')
        
        # Get appropriate template
        templates = self.response_templates['algorithm_request'].get(project_type, 
                    self.response_templates['algorithm_request']['classification'])
        
        intro = random.choice(templates)
        
        # Get algorithm recommendations
        algorithms = self._get_algorithm_recommendations(project_type, data_size, context)
        
        # Build response
        response = f"{intro}\n\n"
        
        for i, algo in enumerate(algorithms[:3], 1):
            response += f"**{i}. {algo['name']}**\n"
            response += f"   • Güven Skoru: {algo['confidence']:.1f}/5.0\n"
            response += f"   • {algo['description']}\n\n"
        
        response += "Bu algoritmaların hangisi hakkında daha fazla bilgi almak istersiniz?"
        
        suggestions = [f"{algo['name']} hakkında bilgi ver" for algo in algorithms[:3]]
        suggestions.append("Kod örneği göster")
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _handle_alternative_request(self, user_message: str, context: Dict) -> Dict:
        """Handle requests for alternative algorithms"""
        # Get discussed algorithms from context
        discussed_algorithms = context.get('conversation_context', {}).get('discussed_algorithms', [])
        
        intro = random.choice(self.response_templates['alternative_requests'])
        
        project_type = context.get('project_type', 'classification')
        data_size = context.get('data_size', 'medium')
        
        # Get alternative algorithms (excluding discussed ones)
        all_algorithms = self._get_algorithm_recommendations(project_type, data_size, context)
        alternatives = [algo for algo in all_algorithms if algo['name'].lower() not in discussed_algorithms]
        
        if not alternatives:
            alternatives = all_algorithms  # If no alternatives, show all
        
        response = f"{intro}\n\n"
        response += "**Alternatif Algoritma Önerileri:**\n\n"
        
        for i, algo in enumerate(alternatives[:3], 1):
            response += f"**{i}. {algo['name']}**\n"
            response += f"   • Güven Skoru: {algo['confidence']:.1f}/5.0\n"
            response += f"   • {algo['description']}\n\n"
        
        response += "Bu alternatiflerden hangisi ilginizi çekiyor?"
        
        suggestions = [f"{algo['name']} tercih ediyorum" for algo in alternatives[:3]]
        suggestions.append("Performans karşılaştırması yap")
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _handle_algorithm_explanation(self, user_message: str, context: Dict) -> Dict:
        """Handle algorithm explanation requests"""
        # Detect which algorithm user is asking about
        algorithm = self._detect_algorithm_in_message(user_message)
        
        if algorithm and algorithm in self.response_templates['algorithm_explanation']:
            algo_info = self.response_templates['algorithm_explanation'][algorithm]
            
            response = f"**{algorithm.title()} Algoritması Hakkında:**\n\n"
            response += f"{algo_info['intro']}\n\n"
            
            response += "**Avantajları:**\n"
            for strength in algo_info['strengths']:
                response += f"• {strength}\n"
            
            response += "\n**Dezavantajları:**\n"
            for weakness in algo_info['weaknesses']:
                response += f"• {weakness}\n"
            
            response += "\n**Kullanım Alanları:**\n"
            for use_case in algo_info['use_cases']:
                response += f"• {use_case}\n"
            
            suggestions = [
                f"{algorithm.title()} kod örneği",
                f"Başka algoritma öner",
                f"{algorithm.title()} vs diğer algoritmalar"
            ]
        else:
            response = "Hangi algoritma hakkında bilgi almak istiyorsunuz?\n\n"
            response += "**Popüler Seçenekler:**\n"
            response += "• XGBoost - Güçlü gradient boosting\n"
            response += "• Random Forest - Güvenilir ensemble method\n"
            response += "• K-Means - Etkili clustering algoritması\n"
            response += "• Neural Networks - Derin öğrenme\n"
            
            suggestions = [
                "XGBoost açıkla",
                "Random Forest nedir?",
                "K-Means nasıl çalışır?",
                "Neural Networks anlat"
            ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _handle_consultation(self, user_message: str, context: Dict) -> Dict:
        """Handle consultation when context is incomplete"""
        missing_info = []
        
        if not context.get('project_type'):
            missing_info.append('project_type_unknown')
        elif not context.get('data_size'):
            missing_info.append('data_size_unknown')
        elif not context.get('data_type'):
            missing_info.append('data_type_unknown')
        
        if missing_info:
            question_type = missing_info[0]
            questions = self.response_templates['consultation_questions'][question_type]
            question = random.choice(questions)
            
            # Generate contextual suggestions
            if question_type == 'project_type_unknown':
                suggestions = [
                    "Sınıflandırma projesi yapıyorum",
                    "Regresyon analizi yapmak istiyorum",
                    "Veri kümeleme yapacağım",
                    "Anomali tespiti yapmak istiyorum"
                ]
            elif question_type == 'data_size_unknown':
                suggestions = [
                    "Küçük veri setim var (< 1000 kayıt)",
                    "Orta boyutta veri setim var (1K-10K)",
                    "Büyük veri setim var (> 10K kayıt)",
                    "Çok büyük veri setim var (> 100K)"
                ]
            else:  # data_type_unknown
                suggestions = [
                    "Sayısal verilerim var",
                    "Kategorik verilerim var",
                    "Metin verileri ile çalışıyorum",
                    "Görüntü verileri kullanıyorum"
                ]
            
            return {
                "response": question,
                "suggestions": suggestions,
                "success": True,
                "ai_powered": False,
                "context_aware": True
            }
        
        # If we have all info, proceed with recommendation
        return self._handle_algorithm_request(user_message, context)
    
    def _handle_comparison_request(self, user_message: str, context: Dict) -> Dict:
        """Handle algorithm comparison requests"""
        project_type = context.get('project_type', 'classification')
        
        response = f"**{project_type.title()} Algoritmaları Karşılaştırması:**\n\n"
        
        algorithms = self._get_algorithm_recommendations(project_type, context.get('data_size', 'medium'), context)
        
        for algo in algorithms[:3]:
            response += f"**{algo['name']}**\n"
            response += f"• Güven Skoru: {algo['confidence']:.1f}/5.0\n"
            response += f"• {algo['description']}\n"
            response += f"• Önerilen Veri Boyutu: {algo['recommended_size']}\n\n"
        
        response += "Hangi algoritmaları daha detaylı karşılaştırmak istersiniz?"
        
        suggestions = [
            f"{algorithms[0]['name']} vs {algorithms[1]['name']}",
            "Performans metrikleri göster",
            "Kod örnekleri ver",
            "Hangi durumda hangisini kullanmalıyım?"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _handle_code_request(self, user_message: str, context: Dict) -> Dict:
        """Handle code example requests"""
        algorithm = self._detect_algorithm_in_message(user_message)
        project_type = context.get('project_type', 'classification')
        
        if algorithm:
            response = f"**{algorithm.title()} Kod Örneği:**\n\n"
            response += f"```python\n"
            response += self._generate_code_example(algorithm, project_type)
            response += f"```\n\n"
            response += f"Bu kod örneği {project_type} problemi için {algorithm} algoritmasını kullanıyor.\n"
            response += "Başka bir algoritma için kod örneği ister misiniz?"
        else:
            response = "Hangi algoritma için kod örneği istiyorsunuz?\n\n"
            response += "**Mevcut Seçenekler:**\n"
            response += "• XGBoost\n• Random Forest\n• K-Means\n• Logistic Regression\n"
        
        suggestions = [
            "XGBoost kod örneği",
            "Random Forest kod örneği",
            "K-Means kod örneği",
            "Başka algoritma öner"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _handle_general_inquiry(self, user_message: str, context: Dict) -> Dict:
        """Handle general inquiries"""
        responses = [
            "Size nasıl yardımcı olabilirim? Makine öğrenmesi projeniz hakkında konuşalım!",
            "Projenizle ilgili hangi konuda yardıma ihtiyacınız var?",
            "Hangi algoritma konusunda bilgi almak istersiniz?",
            "Veri analizinizde hangi aşamada yardım edebilirim?"
        ]
        
        response = random.choice(responses)
        
        suggestions = [
            "Algoritma önerisi istiyorum",
            "Kod örneği göster",
            "Algoritmaları karşılaştır",
            "Projem hakkında soru sor"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "success": True,
            "ai_powered": False,
            "context_aware": True
        }
    
    def _get_algorithm_recommendations(self, project_type: str, data_size: str, context: Dict) -> List[Dict]:
        """Get algorithm recommendations based on context"""
        size_key = 'small_data' if data_size == 'small' else 'large_data' if data_size == 'large' else 'medium_data'
        
        algorithms = self.algorithm_database.get(project_type, {}).get(size_key, [])
        
        recommendations = []
        for i, algo in enumerate(algorithms):
            confidence = 4.5 - (i * 0.3)  # Decreasing confidence
            
            rec = {
                'name': algo,
                'confidence': max(3.0, confidence),
                'description': self._get_algorithm_description(algo, project_type),
                'recommended_size': data_size
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _get_algorithm_description(self, algorithm: str, project_type: str) -> str:
        """Get algorithm description"""
        descriptions = {
            'XGBoost': f"Güçlü gradient boosting algoritması, {project_type} için mükemmel",
            'Random Forest': f"Güvenilir ensemble method, {project_type} için ideal",
            'Logistic Regression': f"Basit ve etkili {project_type} algoritması",
            'K-Means': "Popüler kümeleme algoritması, hızlı ve etkili",
            'Neural Networks': f"Derin öğrenme ile {project_type} için güçlü çözüm",
            'SVM': f"Support Vector Machine, {project_type} için matematiksel yaklaşım"
        }
        
        return descriptions.get(algorithm, f"{algorithm} algoritması")
    
    def _detect_algorithm_in_message(self, message: str) -> str:
        """Detect which algorithm user is asking about"""
        text_lower = message.lower()
        
        algorithms = {
            'xgboost': ['xgboost', 'xgb'],
            'random_forest': ['random forest', 'rf'],
            'k-means': ['k-means', 'kmeans'],
            'logistic_regression': ['logistic regression', 'logistic'],
            'neural_networks': ['neural network', 'neural', 'deep learning'],
            'svm': ['svm', 'support vector']
        }
        
        for algo_key, keywords in algorithms.items():
            if any(keyword in text_lower for keyword in keywords):
                return algo_key
        
        return None
    
    def _generate_code_example(self, algorithm: str, project_type: str) -> str:
        """Generate code example for algorithm"""
        code_templates = {
            'xgboost': f"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri hazırlığı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model = xgb.XGBClassifier(random_state=42)

# Eğitim
model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Doğruluk: {{accuracy:.2f}}")
""",
            'random_forest': f"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Veri hazırlığı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Eğitim
model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Doğruluk: {{accuracy:.2f}}")
""",
            'k-means': """
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Model oluşturma
kmeans = KMeans(n_clusters=3, random_state=42)

# Kümeleme
clusters = kmeans.fit_predict(X)

# Görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.title('K-Means Kümeleme Sonuçları')
plt.show()
"""
        }
        
        return code_templates.get(algorithm, "# Kod örneği hazırlanıyor...")
    
    def _get_safe_fallback(self, user_message: str) -> Dict:
        """Safe fallback when all else fails"""
        return {
            "response": "Üzgünüm, şu anda tam olarak anlayamadım. Makine öğrenmesi projeniz hakkında daha spesifik bilgi verebilir misiniz?",
            "suggestions": [
                "Algoritma önerisi istiyorum",
                "Sınıflandırma projesi yapıyorum",
                "Kod örneği göster",
                "Baştan başlayalım"
            ],
            "success": True,
            "ai_powered": False,
            "context_aware": False
        }

# Global fallback system instance
context_aware_fallback = ContextAwareFallback() 