"""
Response Generation Service - Separated from OpenAIService for better architecture
"""

import logging
import random
from typing import Dict, List, Optional
from services.algorithm_recommender import AlgorithmRecommender

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Generates AI responses with templates and smart suggestions
    """
    
    def __init__(self, algorithm_recommender: AlgorithmRecommender):
        self.algorithm_recommender = algorithm_recommender
        
        # Response templates for different scenarios
        self.response_templates = {
            'greeting': [
                "Merhaba! Ben AlgoMentor, sizin kişisel makine öğrenmesi danışmanınızım. Projeniz hakkında konuşmaya hazırım!",
                "Selam! Size nasıl yardımcı olabilirim? Hangi tür bir ML projesi üzerinde çalışıyorsunuz?",
                "Hey! Makine öğrenmesi dünyasında size rehberlik etmek için buradayım. Projenizden bahseder misiniz?"
            ],
            'consultation': [
                "Anladım! Bu gerçekten ilginç bir proje. Daha fazla detay verebilir misiniz?",
                "Harika! Bu konuda size yardımcı olabilirim. Biraz daha bilgi alabilir miyim?",
                "Süper! Projenizin detaylarını öğrenmek istiyorum. Biraz daha açıklayabilir misiniz?"
            ],
            'recommendation_intro': [
                "Mükemmel! Verdiğiniz bilgilere göre size en uygun algoritmaları öneriyorum:",
                "Harika! Projeniz için ideal algoritmaları buldum. İşte önerilerim:",
                "Süper! Size özel hazırladığım algoritma önerileri:"
            ]
        }
    
    def generate_consultation_response(self, user_message: str, context: Dict) -> Dict:
        """Generate consultation response based on context"""
        missing_info = self._identify_missing_information(context)
        
        if not missing_info:
            # We have enough info, suggest recommendations
            return {
                "response": "Mükemmel! Projeniz hakkında yeterli bilgiye sahibim. Size en uygun algoritmaları önerebilirim. Önerilerimi görmek ister misiniz?",
                "suggestions": [
                    "Evet, önerilerini görmek istiyorum",
                    "Önce biraz daha bilgi almak istiyorum",
                    "Hangi kriterlere göre öneride bulunuyorsun?"
                ],
                "success": True
            }
        
        # Generate question based on missing info
        question = self._generate_consultation_question(missing_info[0], context)
        suggestions = self._generate_consultation_suggestions(missing_info, context)
        
        return {
            "response": question,
            "suggestions": suggestions,
            "success": True
        }
    
    def _identify_missing_information(self, context: Dict) -> List[str]:
        """Identify what information is still missing"""
        missing = []
        
        required_fields = {
            'project_type': 'Problem türü',
            'data_size': 'Veri büyüklüğü',
            'data_type': 'Veri türü'
        }
        
        for field, description in required_fields.items():
            if not context.get(field):
                missing.append(field)
        
        return missing
    
    def _generate_consultation_question(self, missing_field: str, context: Dict) -> str:
        """Generate appropriate question for missing information"""
        questions = {
            'project_type': [
                "Harika! Şimdi projenizin türü hakkında konuşalım. Hangi tür bir problem çözmeye çalışıyorsunuz?",
                "Süper! Projenizin amacı nedir? Sınıflandırma, regresyon, kümeleme gibi hangi kategoride?",
                "Mükemmel! Projenizde ne yapmaya çalışıyorsunuz? Veri analizi, tahmin, gruplama?"
            ],
            'data_size': [
                "Anladım! Peki veri setinizin büyüklüğü nasıl? Kaç satır veri ile çalışıyorsunuz?",
                "Harika! Verileriniz hakkında konuşalım. Veri setiniz ne kadar büyük?",
                "Süper! Kaç tane veri noktanız var? Küçük, orta, büyük ölçekte mi?"
            ],
            'data_type': [
                "Mükemmel! Verilerinizin türü nedir? Sayısal, kategorik, metin, görüntü?",
                "Harika! Hangi tür veri ile çalışıyorsunuz? Sayılar, kelimeler, resimler?",
                "Süper! Verilerinizin formatı nasıl? Tablo, metin, görsel veri mi?"
            ]
        }
        
        return random.choice(questions.get(missing_field, ["Biraz daha detay verebilir misiniz?"]))
    
    def _generate_consultation_suggestions(self, missing_info: List[str], context: Dict) -> List[str]:
        """Generate smart suggestions based on missing information"""
        suggestions = []
        
        if 'project_type' in missing_info:
            suggestions.extend([
                "Sınıflandırma yapıyorum",
                "Regresyon analizi yapmak istiyorum",
                "Kümeleme analizi yapacağım",
                "Anomali tespiti yapıyorum"
            ])
        
        if 'data_size' in missing_info:
            suggestions.extend([
                "Küçük veri setim var (< 1000 satır)",
                "Orta büyüklükte veri setim var",
                "Büyük veri setim var (> 100k satır)"
            ])
        
        if 'data_type' in missing_info:
            suggestions.extend([
                "Sayısal verilerle çalışıyorum",
                "Kategorik verilerim var",
                "Metin verisi analiz ediyorum",
                "Görüntü verisi kullanıyorum"
            ])
        
        # Keep only top 4 suggestions
        return suggestions[:4]
    
    def generate_algorithm_recommendations(self, context: Dict) -> Dict:
        """Generate algorithm recommendations with explanations"""
        try:
            # Get recommendations from algorithm recommender
            recommendations = self.algorithm_recommender.get_recommendations(
                project_type=context.get('project_type'),
                data_size=context.get('data_size'),
                data_type=context.get('data_type'),
                complexity_preference=context.get('complexity', 'medium')
            )
            
            if not recommendations:
                return self._generate_fallback_recommendations(context)
            
            # Format recommendations with explanations
            response_parts = []
            response_parts.append(random.choice(self.response_templates['recommendation_intro']))
            response_parts.append("")
            
            for i, rec in enumerate(recommendations[:3], 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                response_parts.append(f"{emoji} **{rec['algorithm']}** (Güven: {rec['confidence_score']:.0%})")
                response_parts.append(f"   ✅ {rec['description']}")
                
                if rec.get('match_reasons'):
                    response_parts.append(f"   💡 {rec['match_reasons'][0]}")
                
                response_parts.append("")
            
            response_parts.append("Hangi algoritma hakkında daha fazla bilgi almak istersiniz?")
            
            # Generate suggestions
            suggestions = []
            for rec in recommendations[:3]:
                suggestions.append(f"{rec['algorithm']} hakkında bilgi ver")
            suggestions.append("Kod örneği göster")
            
            return {
                "response": "\n".join(response_parts),
                "suggestions": suggestions,
                "recommendations": recommendations,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._generate_fallback_recommendations(context)
    
    def _generate_fallback_recommendations(self, context: Dict) -> Dict:
        """Generate fallback recommendations when main system fails"""
        project_type = context.get('project_type', '').lower()
        
        fallback_recommendations = {
            'classification': [
                {'algorithm': 'Random Forest', 'confidence_score': 0.85, 'description': 'Güvenilir ve etkili sınıflandırma algoritması'},
                {'algorithm': 'Logistic Regression', 'confidence_score': 0.75, 'description': 'Basit ve yorumlanabilir algoritma'},
                {'algorithm': 'SVM', 'confidence_score': 0.70, 'description': 'Güçlü sınıflandırma performansı'}
            ],
            'regression': [
                {'algorithm': 'Linear Regression', 'confidence_score': 0.80, 'description': 'Basit ve etkili regresyon algoritması'},
                {'algorithm': 'Random Forest Regressor', 'confidence_score': 0.85, 'description': 'Güvenilir regresyon performansı'},
                {'algorithm': 'XGBoost', 'confidence_score': 0.90, 'description': 'Yüksek performanslı gradient boosting'}
            ],
            'clustering': [
                {'algorithm': 'K-Means', 'confidence_score': 0.85, 'description': 'Popüler ve etkili kümeleme algoritması'},
                {'algorithm': 'DBSCAN', 'confidence_score': 0.75, 'description': 'Yoğunluk tabanlı kümeleme'},
                {'algorithm': 'Hierarchical Clustering', 'confidence_score': 0.70, 'description': 'Dendogram ile görselleştirme'}
            ]
        }
        
        # Select appropriate recommendations
        if 'sınıflandırma' in project_type or 'classification' in project_type:
            recommendations = fallback_recommendations['classification']
        elif 'regresyon' in project_type or 'regression' in project_type:
            recommendations = fallback_recommendations['regression']
        elif 'kümeleme' in project_type or 'clustering' in project_type:
            recommendations = fallback_recommendations['clustering']
        else:
            recommendations = fallback_recommendations['classification']  # Default
        
        response_parts = [
            "Projeniz için temel algoritma önerilerim:",
            ""
        ]
        
        for i, rec in enumerate(recommendations, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            response_parts.append(f"{emoji} **{rec['algorithm']}** (Güven: {rec['confidence_score']:.0%})")
            response_parts.append(f"   ✅ {rec['description']}")
            response_parts.append("")
        
        return {
            "response": "\n".join(response_parts),
            "suggestions": [
                f"{rec['algorithm']} hakkında bilgi ver" for rec in recommendations
            ],
            "recommendations": recommendations,
            "success": True
        }
    
    def generate_algorithm_explanation(self, algorithm: str, context: Dict) -> Dict:
        """Generate detailed algorithm explanation"""
        explanations = {
            'random forest': {
                'description': 'Random Forest, birçok karar ağacını birleştiren güçlü bir ensemble algoritmasıdır.',
                'advantages': ['Overfitting\'e karşı dirençli', 'Özellik önemini gösterir', 'Hem sınıflandırma hem regresyon için kullanılır'],
                'disadvantages': ['Yorumlanması zor', 'Büyük veri setlerinde yavaş olabilir'],
                'use_cases': ['Özellik seçimi', 'Genel amaçlı sınıflandırma', 'Regresyon problemleri']
            },
            'xgboost': {
                'description': 'XGBoost, gradient boosting algoritmasının optimize edilmiş versiyonudur.',
                'advantages': ['Yüksek performans', 'Hızlı training', 'Regularization desteği'],
                'disadvantages': ['Hiperparametre tuning gerektirir', 'Overfitting riski'],
                'use_cases': ['Kaggle yarışmaları', 'Structured data problems', 'Ranking problemleri']
            },
            'logistic regression': {
                'description': 'Logistic Regression, sınıflandırma için kullanılan basit ve etkili bir algoritmadır.',
                'advantages': ['Yorumlanabilir', 'Hızlı', 'Probabilistic output'],
                'disadvantages': ['Linear relationships varsayar', 'Outlier\'lara hassas'],
                'use_cases': ['Binary classification', 'Probability estimation', 'Baseline model']
            }
        }
        
        algo_key = algorithm.lower()
        if algo_key in explanations:
            info = explanations[algo_key]
            
            response_parts = [
                f"🧠 **{algorithm}** hakkında detaylı bilgi:",
                "",
                f"**Açıklama:** {info['description']}",
                "",
                "**Avantajları:**"
            ]
            
            for advantage in info['advantages']:
                response_parts.append(f"✅ {advantage}")
            
            response_parts.extend([
                "",
                "**Dezavantajları:"
            ])
            
            for disadvantage in info['disadvantages']:
                response_parts.append(f"❌ {disadvantage}")
            
            response_parts.extend([
                "",
                "**Kullanım Alanları:**"
            ])
            
            for use_case in info['use_cases']:
                response_parts.append(f"🎯 {use_case}")
            
            return {
                "response": "\n".join(response_parts),
                "suggestions": [
                    f"{algorithm} kod örneği göster",
                    "Başka algoritma öner",
                    "Performans karşılaştırması yap",
                    "Implementasyon detayları ver"
                ],
                "success": True
            }
        else:
            return {
                "response": f"Üzgünüm, {algorithm} hakkında detaylı bilgim şu anda mevcut değil. Başka bir algoritma hakkında bilgi almak ister misiniz?",
                "suggestions": [
                    "Random Forest hakkında bilgi ver",
                    "XGBoost açıkla",
                    "Logistic Regression anlat",
                    "Başka algoritma öner"
                ],
                "success": False
            } 