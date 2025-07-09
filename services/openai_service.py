import os
from openai import OpenAI
from typing import Dict, List, Optional
from services.algorithm_recommender import AlgorithmRecommender
from dotenv import load_dotenv
import random

# Load environment variables from .env file
load_dotenv()

class OpenAIService:
    def __init__(self):
        """
        Advanced Hybrid AI-powered algorithm consultant with intelligent fallback
        """
        self.algorithm_recommender = AlgorithmRecommender()
        
        # Initialize OpenAI with modern client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            try:
                self.openai_client = OpenAI(api_key=api_key)
                # Test the connection
                self.openai_client.models.list()
                self.openai_enabled = True
                print("✅ OpenAI API successfully initialized and tested")
            except Exception as e:
                print(f"⚠️ OpenAI API issue (quota/connection): {str(e)[:100]}...")
                self.openai_enabled = False
                self.openai_client = None
        else:
            self.openai_enabled = False
            self.openai_client = None
            print("⚠️ OpenAI API key not found, using advanced fallback system")
        
        # Always use our advanced AI system regardless of OpenAI status
        self.use_advanced_ai = True
        print("🤖 Advanced AI Algorithm Consultant initialized with intelligent conversation engine")
        
        # System prompts for different scenarios
        self.algorithm_expert_prompt = """Sen deneyimli bir makine öğrenmesi algoritma uzmanısın. Kullanıcılara algoritma önerileri verirken:

1. Paragraf halinde, akıcı ve detaylı açıklamalar yap
2. Teknik terimleri açıkla ama anlaşılır tut
3. Gerçek dünya örnekleri ver
4. Avantaj ve dezavantajları dengeli şekilde açıkla
5. Pratik uygulama ipuçları ekle
6. Türkçe konuş ve dostane bir ton kullan

Algoritma veri setinde şu kodlama sistemi kullanılıyor:
- Öğrenme türü: sl (Supervised), ul (Unsupervised), ssl (Semi-supervised), rl (Reinforcement)
- Kullanım alanı: cl (Classification), re (Regression), ts (Time Series), np (NLP), vb.
- Karmaşıklık: comp1-comp5 (1=basit, 5=çok karmaşık)
- Veri büyüklüğü: MB, MB-GB, GB, GB-TB, TB
- Popülerlik: p1-p5 (1=az popüler, 5=çok popüler)

Her zaman paragraf şeklinde detaylı açıklamalar yap."""

        self.consultation_prompt = """Sen bir makine öğrenmesi danışmanısın. Kullanıcıların proje gereksinimlerini anlamaya çalışıyorsun. 

Görevin:
1. Eksik bilgileri nazikçe sor
2. Paragraf halinde, samimi ve yönlendirici konuş
3. Kullanıcının seviyesine uygun açıklamalar yap
4. Projenin hedefini net anlayıp doğru yönlendir
5. Türkçe konuş ve dostane ol

Kısa listeler yerine akıcı paragraflar halinde konuş."""
    
    def get_chat_response(self, user_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
        """
        Hybrid AI response: OpenAI + Custom Algorithm Model
        """
        try:
            # Handle None or empty messages
            if user_message is None:
                user_message = ""
            
            print(f"\n🔍 Processing: '{user_message}'")
            
            # Analyze project context from conversation
            project_context = self._extract_project_context(user_message, conversation_history)
            print(f"📊 Project Context: {project_context}")
            
            # Check if user is asking algorithm-specific questions (don't restart consultation)
            if self._is_algorithm_specific_question(user_message, project_context):
                return self._handle_algorithm_question(user_message, project_context)
            
            # Check if we have enough info for algorithm recommendation
            if self._should_recommend_algorithms(project_context):
                return self._generate_algorithm_recommendations(user_message, project_context)
            else:
                return self._generate_consultation_response(user_message, project_context)
                
        except Exception as e:
            print(f"❌ Error in AI service: {str(e)}")
            return self._get_emergency_fallback()
    
    def _extract_project_context(self, current_message: str, conversation_history: Optional[List[Dict]] = None) -> Dict:
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
        elif any(word in user_msg_lower for word in ['detay', 'açıkla', 'nedir', 'nasıl çalışır', 'ne yapar', 'avantaj', 'dezavantaj', 'ne zaman kullan']):
            return self._generate_algorithm_explanation(user_message, context)
        
        # Default: generate new recommendations
        else:
            return self._generate_algorithm_recommendations(user_message, context)
    
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
            model="gpt-4o",  # Use the latest GPT-4 Omni model for best quality
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
        
        for algo, explanation in explanations.items():
            if algo in user_msg_lower:
                return {
                    "response": explanation,
                    "suggestions": [
                        f"{algo.title()} kod örneği",
                        "Hiperparametre ayarları",
                        "Diğer algoritmalarla karşılaştır"
                    ],
                    "success": True
                }
        
        # Generic algorithm explanation
        return {
            "response": "🤖 Hangi algoritma hakkında bilgi almak istiyorsunuz? Size detaylarını açıklayabilirim.",
            "suggestions": [
                "XGBoost nedir?",
                "Random Forest açıkla",
                "SVM nasıl çalışır?"
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
                complexity_preference=context.get('complexity', 'medium')
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
Veri boyutu: {context.get('data_size', 'Orta')}
Veri türü: {context.get('data_type', 'Sayısal')}
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
                model="gpt-4o",  # Use GPT-4 Omni for superior algorithmic advice and detailed explanations
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
            "Kod örneği"
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
                'classification': "🧠 Beyin yapısını taklit eder. Çok karmaşık ilişkileri öğrenebilir.",
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
- Proje türü: {context.get('project_type', 'Belirsiz')}
- Veri boyutu: {context.get('data_size', 'Belirsiz')}
- Veri türü: {context.get('data_type', 'Belirsiz')}
- Sınıf sayısı: {context.get('class_count', 'Belirsiz')}

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
2. Eksik bilgi varsa nazikçe sor ama zorlama
3. Projesini anlayıp doğru yönlendir
4. Teknik terimlerden kaçın, sade konuş
5. 2-3 paragraf halinde cevap ver

Kısa listeler yerine akıcı konuşma yap.
"""}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",  # Premium GPT-4 for consultation responses
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
                'general': "Derin öğrenmenin kapısını açan temel algoritma. Karmaşık problemlerde çok güçlü ama yeterli veri gerektirir."
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
        Emergency response when everything fails
        """
        return {
            "response": "Üzgünüm, şu anda teknik bir sorun yaşıyorum. Projeniz hakkında daha basit terimlerle anlatabilir misiniz?",
            "suggestions": [
                "Hangi tür veri analizi yapacağım?",
                "Ne tür sonuç elde etmek istiyorum?",
                "Elimde ne kadar veri var?"
            ],
            "success": True
        } 