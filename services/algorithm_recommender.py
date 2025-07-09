import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os

class AlgorithmRecommender:
    def __init__(self):
        """
        Algoritma veri setini yükle ve recommendation sistemi başlat
        """
        self.df = None
        self.load_dataset()
        
        # Kodlama sistemi mapping'leri
        self.mappings = {
            'learning_type': {
                'sl': 'Supervised Learning',
                'ul': 'Unsupervised Learning', 
                'ssl': 'Semi-Supervised Learning',
                'rl': 'Reinforcement Learning'
            },
            'use_case': {
                'cl': 'Classification',
                're': 'Regression',
                'ip': 'Image Processing',
                'ts': 'Time Series',
                'np': 'Natural Language Processing',
                'dr': 'Dimensionality Reduction',
                'dap': 'Data Augmentation',
                'vis': 'Visualization',
                'dre': 'Data Exploration',
                'fe': 'Feature Engineering',
                'g': 'Gaming',
                'se': 'Security',
                'ed': 'Education',
                'mol': 'Model Optimization',
                'so': 'Speech/Audio'
            },
            'complexity': {
                'comp1': 'Very Low',
                'comp2': 'Low', 
                'comp3': 'Medium',
                'comp4': 'High',
                'comp5': 'Very High'
            },
            'data_type': {
                'nd': 'Numerical Data',
                'cd': 'Categorical Data',
                'tsd': 'Time Series Data',
                'ad': 'Audio Data',
                'id': 'Image Data',
                'nod': 'No Specific Data',
                'td': 'Text Data'
            },
            'data_size': {
                'MB': 'Small (MB)',
                'MB-GB': 'Medium (MB-GB)',
                'GB': 'Large (GB)',
                'GB-TB': 'Very Large (GB-TB)',
                'TB': 'Massive (TB)'
            },
            'popularity': {
                'p1': 'Very Low',
                'p2': 'Low',
                'p3': 'Medium', 
                'p4': 'High',
                'p5': 'Very High'
            }
        }
    
    def load_dataset(self):
        """
        Algoritma veri setini yükle
        """
        try:
            dataset_path = os.path.join('algorithms', 'Veri_seti.csv')
            if os.path.exists(dataset_path):
                self.df = pd.read_csv(dataset_path)
                print(f"Dataset loaded successfully: {len(self.df)} algorithms")
            else:
                print(f"Dataset not found at {dataset_path}")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
    
    def get_recommendations(self, 
                          project_type: str = None,
                          data_size: str = None, 
                          data_type: str = None,
                          complexity_preference: str = None,
                          top_n: int = 5) -> List[Dict]:
        """
        Kullanıcı gereksinimlerine göre algoritma önerileri döndür
        """
        if self.df is None:
            return self._get_fallback_recommendations(project_type)
        
        try:
            # Filtreleme başlat
            filtered_df = self.df.copy()
            
            # Proje türüne göre filtrele
            if project_type:
                if 'classification' in project_type.lower() or 'sınıflandırma' in project_type.lower():
                    # Sınıflandırma için supervised learning + 'cl' kullanım alanı
                    filtered_df = filtered_df[
                        (filtered_df['Öğrenme Türü'] == 'sl') & 
                        (filtered_df['Kullanım Alanı'].str.contains('cl', na=False))
                    ]
                elif 'regression' in project_type.lower() or 'regresyon' in project_type.lower():
                    # Regresyon için supervised learning + 're' kullanım alanı
                    filtered_df = filtered_df[
                        (filtered_df['Öğrenme Türü'] == 'sl') & 
                        (filtered_df['Kullanım Alanı'].str.contains('re', na=False))
                    ]
                elif 'clustering' in project_type.lower() or 'kümeleme' in project_type.lower():
                    # Kümeleme için unsupervised learning + 'cl' kullanım alanı
                    filtered_df = filtered_df[
                        (filtered_df['Öğrenme Türü'] == 'ul') & 
                        (filtered_df['Kullanım Alanı'].str.contains('cl', na=False))
                    ]
                elif 'time_series' in project_type.lower() or 'zaman serisi' in project_type.lower():
                    filtered_df = filtered_df[filtered_df['Kullanım Alanı'].str.contains('ts', na=False)]
                elif 'anomaly' in project_type.lower() or 'anomali' in project_type.lower():
                    # Anomali tespiti için unsupervised learning
                    filtered_df = filtered_df[filtered_df['Öğrenme Türü'] == 'ul']
                elif 'nlp' in project_type.lower() or 'doğal dil' in project_type.lower():
                    filtered_df = filtered_df[filtered_df['Kullanım Alanı'].str.contains('np', na=False)]
            
            # Veri boyutuna göre filtrele
            if data_size:
                if 'small' in data_size.lower() or '1000' in data_size or 'küçük' in data_size.lower():
                    filtered_df = filtered_df[filtered_df['Veri Büyüklüğü '].isin(['MB', 'MB-GB'])]
                elif 'large' in data_size.lower() or '10000' in data_size or 'büyük' in data_size.lower():
                    filtered_df = filtered_df[filtered_df['Veri Büyüklüğü '].isin(['GB', 'GB-TB', 'TB'])]
            
            # Veri tipine göre filtrele
            if data_type:
                if 'numerical' in data_type.lower() or 'sayısal' in data_type.lower():
                    filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('nd', na=False)]
                elif 'text' in data_type.lower() or 'metin' in data_type.lower():
                    filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('td', na=False)]
                elif 'categorical' in data_type.lower() or 'kategorik' in data_type.lower():
                    filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('cd', na=False)]
                elif 'image' in data_type.lower() or 'görüntü' in data_type.lower():
                    filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('id', na=False)]
            
            # Karmaşıklık seviyesine göre sırala (düşükten yükseğe)
            complexity_order = ['comp1', 'comp2', 'comp3', 'comp4', 'comp5']
            filtered_df['complexity_rank'] = filtered_df['Karmaşıklık Düzeyi'].map(
                {comp: idx for idx, comp in enumerate(complexity_order)}
            )
            
            # Popülerlik ve karmaşıklık kombinasyonu ile sırala
            popularity_order = ['p5', 'p4', 'p3', 'p2', 'p1']
            filtered_df['popularity_rank'] = filtered_df['Popülerlik'].map(
                {pop: idx for idx, pop in enumerate(popularity_order)}
            )
            
            # Sıralama: önce popülerlik, sonra düşük karmaşıklık
            filtered_df = filtered_df.sort_values(['popularity_rank', 'complexity_rank'])
            
            # Top N algoritma seç
            top_algorithms = filtered_df.head(top_n)
            
            # Sonuçları formatla
            recommendations = []
            for _, row in top_algorithms.iterrows():
                confidence_score = self._calculate_confidence_score(row)
                algorithm_name = row['Algoritma Adı']
                complexity = self.mappings['complexity'].get(row['Karmaşıklık Düzeyi'], 'Medium')
                
                rec = {
                    'algorithm': algorithm_name,  # 'name' yerine 'algorithm'
                    'confidence_score': confidence_score,
                    'learning_type': self.mappings['learning_type'].get(row['Öğrenme Türü'], row['Öğrenme Türü']),
                    'complexity': complexity,
                    'popularity': self.mappings['popularity'].get(row['Popülerlik'], row['Popülerlik']),
                    'data_size_support': self.mappings['data_size'].get(row['Veri Büyüklüğü '], row['Veri Büyüklüğü ']),
                    'hardware_requirement': row['Donanım Gerkesinimleri'],
                    'use_cases': row['Kullanım Alanı'],
                    'data_types': row['Veri Tipi'],
                    'description': f"{algorithm_name} - {complexity} complexity algorithm"
                }
                recommendations.append(rec)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {str(e)}")
            return self._get_fallback_recommendations(project_type)
    
    def _get_fallback_recommendations(self, project_type: str = None) -> List[Dict]:
        """
        Veri seti yüklenemediğinde fallback öneriler
        """
        if project_type and 'classification' in project_type.lower():
            return [
                {
                    'algorithm': 'Logistic Regression',
                    'confidence_score': 4.2,
                    'learning_type': 'Supervised Learning',
                    'complexity': 'Low',
                    'popularity': 'High',
                    'data_size_support': 'Small to Medium',
                    'hardware_requirement': 0.0,
                    'description': 'Fast, interpretable, good baseline for classification'
                },
                {
                    'algorithm': 'Random Forest',
                    'confidence_score': 4.8,
                    'learning_type': 'Supervised Learning', 
                    'complexity': 'Medium',
                    'popularity': 'Very High',
                    'data_size_support': 'Medium to Large',
                    'hardware_requirement': 0.5,
                    'description': 'Robust, handles overfitting well, excellent for classification'
                },
                {
                    'algorithm': 'Support Vector Machine',
                    'confidence_score': 4.0,
                    'learning_type': 'Supervised Learning',
                    'complexity': 'Medium',
                    'popularity': 'High',
                    'data_size_support': 'Small to Medium',
                    'hardware_requirement': 0.3,
                    'description': 'High accuracy, works well with small datasets'
                }
            ]
        elif project_type and 'regression' in project_type.lower():
            return [
                {
                    'algorithm': 'Linear Regression',
                    'confidence_score': 4.0,
                    'learning_type': 'Supervised Learning',
                    'complexity': 'Low',
                    'popularity': 'Very High',
                    'data_size_support': 'All sizes',
                    'hardware_requirement': 0.1,
                    'description': 'Simple, fast, interpretable regression algorithm'
                },
                {
                    'algorithm': 'Random Forest Regressor',
                    'confidence_score': 4.6,
                    'learning_type': 'Supervised Learning',
                    'complexity': 'Medium',
                    'popularity': 'Very High',
                    'data_size_support': 'Medium to Large',
                    'hardware_requirement': 0.5,
                    'description': 'Robust regression with feature importance'
                },
                {
                    'algorithm': 'XGBoost Regressor',
                    'confidence_score': 4.8,
                    'learning_type': 'Supervised Learning',
                    'complexity': 'High',
                    'popularity': 'Very High',
                    'data_size_support': 'All sizes',
                    'hardware_requirement': 0.7,
                    'description': 'State-of-the-art gradient boosting for regression'
                }
            ]
        
        return [
            {
                'algorithm': 'Random Forest',
                'confidence_score': 4.5,
                'learning_type': 'Supervised Learning',
                'complexity': 'Medium', 
                'popularity': 'Very High',
                'data_size_support': 'Medium to Large',
                'hardware_requirement': 0.5,
                'description': 'Versatile, robust algorithm for various problems'
            }
        ]
    
    def get_algorithm_details(self, algorithm_name: str) -> Dict:
        """
        Belirli bir algoritmanın detaylarını döndür
        """
        if self.df is None:
            return {}
        
        try:
            algorithm_row = self.df[self.df['Algoritma Adı'].str.contains(algorithm_name, case=False, na=False)]
            if not algorithm_row.empty:
                row = algorithm_row.iloc[0]
                return {
                    'name': row['Algoritma Adı'],
                    'learning_type': self.mappings['learning_type'].get(row['Öğrenme Türü'], row['Öğrenme Türü']),
                    'use_cases': row['Kullanım Alanı'],
                    'complexity': self.mappings['complexity'].get(row['Karmaşıklık Düzeyi'], row['Karmaşıklık Düzeyi']),
                    'model_structure': row['Model Yapısı'],
                    'overfitting_tendency': row['Aşırı Öğrenme Eğilimi'],
                    'layer_type': row['Katman Tipi'],
                    'data_types': row['Veri Tipi'],
                    'hardware_requirements': row['Donanım Gerkesinimleri'],
                    'data_size': self.mappings['data_size'].get(row['Veri Büyüklüğü '], row['Veri Büyüklüğü ']),
                    'fine_tune_req': row['FineTune Gereksinimi'],
                    'popularity': self.mappings['popularity'].get(row['Popülerlik'], row['Popülerlik'])
                }
        except Exception as e:
            print(f"Error getting algorithm details: {str(e)}")
        
        return {} 

    def _calculate_confidence_score(self, row) -> float:
        """
        Algoritma için güven skoru hesapla (1-5 arası)
        """
        score = 3.0  # Base score
        
        # Popülerlik bonus
        popularity_bonus = {
            'p5': 1.5, 'p4': 1.0, 'p3': 0.5, 'p2': 0.0, 'p1': -0.5
        }
        score += popularity_bonus.get(row['Popülerlik'], 0)
        
        # Karmaşıklık cezası (yüksek karmaşıklık = düşük puan)
        complexity_penalty = {
            'comp1': 0.5, 'comp2': 0.3, 'comp3': 0.0, 'comp4': -0.3, 'comp5': -0.5
        }
        score += complexity_penalty.get(row['Karmaşıklık Düzeyi'], 0)
        
        # 1-5 arasında sınırla
        return max(1.0, min(5.0, score)) 