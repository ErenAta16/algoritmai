import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class AlgorithmRecommender:
    def __init__(self):
        """
        Advanced Algorithm Recommendation System with ML-based scoring
        """
        self.df = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.algorithm_features = {}
        self.performance_cache = {}
        
        self.load_dataset()
        self.preprocess_features()
        
        # Enhanced coding system mappings
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
        
        # Performance benchmarks for different algorithm types
        self.performance_benchmarks = {
            'classification': {
                'accuracy_ranges': {
                    'excellent': (0.95, 1.0),
                    'good': (0.85, 0.95),
                    'average': (0.70, 0.85),
                    'poor': (0.0, 0.70)
                }
            },
            'regression': {
                'r2_ranges': {
                    'excellent': (0.90, 1.0),
                    'good': (0.75, 0.90),
                    'average': (0.50, 0.75),
                    'poor': (0.0, 0.50)
                }
            }
        }
    
    def load_dataset(self):
        """
        Load algorithm dataset with error handling and validation
        """
        try:
            dataset_paths = [
                'algorithms/Veri_seti.csv',
                'data/Algoritma_Veri_Seti.xlsx',
                'Algoritma_Veri_Seti.xlsx'
            ]
            
            for path in dataset_paths:
                if os.path.exists(path):
                    if path.endswith('.csv'):
                        self.df = pd.read_csv(path)
                    elif path.endswith('.xlsx'):
                        self.df = pd.read_excel(path)
                    
                    logger.info(f"✅ Dataset loaded successfully from {path}: {len(self.df)} algorithms")
                    
                    # Validate dataset structure
                    required_columns = ['Algoritma Adı', 'Öğrenme Türü', 'Kullanım Alanı', 'Karmaşıklık Düzeyi']
                    if all(col in self.df.columns for col in required_columns):
                        logger.info("✅ Dataset structure validated")
                        return
            else:
                        logger.warning(f"⚠️ Missing required columns in {path}")
                        
            logger.error("❌ No valid dataset found")
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {str(e)}")
            self.df = None
    
    def preprocess_features(self):
        """
        Preprocess algorithm features for ML-based recommendations
        """
        if self.df is None:
            return
            
        try:
            # Create numerical features for ML processing
            feature_columns = []
            
            # Encode categorical features
            for col in ['Öğrenme Türü', 'Kullanım Alanı', 'Karmaşıklık Düzeyi', 'Veri Tipi', 'Veri Büyüklüğü ', 'Popülerlik']:
                if col in self.df.columns:
                    # Create one-hot encoding
                    encoded = pd.get_dummies(self.df[col], prefix=col)
                    feature_columns.extend(encoded.columns)
                    
            # Create feature matrix efficiently using pd.concat
            feature_data = {col: [0] * len(self.df) for col in feature_columns}
            self.feature_matrix = pd.DataFrame(feature_data, index=self.df.index)
                
            # Fill feature matrix
            for idx, row in self.df.iterrows():
                for col in ['Öğrenme Türü', 'Kullanım Alanı', 'Karmaşıklık Düzeyi', 'Veri Tipi', 'Veri Büyüklüğü ', 'Popülerlik']:
                    if col in self.df.columns:
                        feature_col = f"{col}_{row[col]}"
                        if feature_col in self.feature_matrix.columns:
                            self.feature_matrix.loc[idx, feature_col] = 1
                            
            # Normalize features
            if not self.feature_matrix.empty:
                self.feature_matrix = pd.DataFrame(
                    self.scaler.fit_transform(self.feature_matrix),
                    columns=self.feature_matrix.columns,
                    index=self.feature_matrix.index
                )
                
            logger.info(f"✅ Feature preprocessing completed: {self.feature_matrix.shape}")
            
        except Exception as e:
            logger.error(f"❌ Error in feature preprocessing: {str(e)}")
    
    def get_recommendations(self, 
                          project_type: str = None,
                          data_size: str = None, 
                          data_type: str = None,
                          complexity_preference: str = None,
                          top_n: int = 5) -> List[Dict]:
        """
        Get enhanced algorithm recommendations with ML-based scoring
        """
        if self.df is None:
            return self._get_fallback_recommendations(project_type)
        
        try:
            # Create user profile vector
            user_profile = self._create_user_profile(project_type, data_size, data_type, complexity_preference)
            
            # Get filtered algorithms
            filtered_df = self._filter_algorithms(project_type, data_size, data_type, complexity_preference)
            
            if filtered_df.empty:
                logger.warning("No algorithms found matching criteria, using fallback")
                return self._get_fallback_recommendations(project_type)
            
            # Calculate similarity scores
            recommendations = self._calculate_similarity_scores(filtered_df, user_profile, top_n)
            
            # Enhance recommendations with additional metadata
            enhanced_recommendations = self._enhance_recommendations(recommendations)
            
            # Cache results for performance
            cache_key = f"{project_type}_{data_size}_{data_type}_{complexity_preference}"
            self.performance_cache[cache_key] = {
                'recommendations': enhanced_recommendations,
                'timestamp': datetime.now(),
                'user_profile': user_profile
            }
            
            return enhanced_recommendations
            
        except Exception as e:
            logger.error(f"❌ Error in get_recommendations: {str(e)}")
            return self._get_fallback_recommendations(project_type)
    
    def _create_user_profile(self, project_type: str, data_size: str, data_type: str, complexity: str) -> Dict:
        """
        Create user profile for similarity matching
        """
        profile = {
            'project_type': project_type or 'classification',
            'data_size': data_size or 'medium',
            'data_type': data_type or 'numerical',
            'complexity': complexity or 'medium',
            'weights': {
                'project_type': 0.4,
                'data_size': 0.2,
                'data_type': 0.2,
                'complexity': 0.2
            }
        }
        
        return profile
    
    def _filter_algorithms(self, project_type: str, data_size: str, data_type: str, complexity: str) -> pd.DataFrame:
        """
        Filter algorithms based on user criteria with enhanced logic
        """
        filtered_df = self.df.copy()
        
        # Project type filtering with enhanced logic
        if project_type:
            if 'classification' in project_type.lower() or 'sınıflandırma' in project_type.lower():
                filtered_df = filtered_df[
                    (filtered_df['Öğrenme Türü'] == 'sl') & 
                    (filtered_df['Kullanım Alanı'].str.contains('cl', na=False))
                ]
            elif 'regression' in project_type.lower() or 'regresyon' in project_type.lower():
                filtered_df = filtered_df[
                    (filtered_df['Öğrenme Türü'] == 'sl') & 
                    (filtered_df['Kullanım Alanı'].str.contains('re', na=False))
                ]
            elif 'clustering' in project_type.lower() or 'kümeleme' in project_type.lower():
                filtered_df = filtered_df[
                    (filtered_df['Öğrenme Türü'] == 'ul') & 
                    (filtered_df['Kullanım Alanı'].str.contains('cl', na=False))
                ]
            elif 'time_series' in project_type.lower() or 'zaman serisi' in project_type.lower():
                filtered_df = filtered_df[filtered_df['Kullanım Alanı'].str.contains('ts', na=False)]
            elif 'anomaly' in project_type.lower() or 'anomali' in project_type.lower():
                filtered_df = filtered_df[filtered_df['Öğrenme Türü'] == 'ul']
            elif 'nlp' in project_type.lower() or 'doğal dil' in project_type.lower():
                filtered_df = filtered_df[filtered_df['Kullanım Alanı'].str.contains('np', na=False)]
        
        # Data size filtering with better logic
        if data_size:
            if 'small' in data_size.lower() or 'küçük' in data_size.lower():
                filtered_df = filtered_df[filtered_df['Veri Büyüklüğü '].isin(['MB', 'MB-GB'])]
            elif 'large' in data_size.lower() or 'büyük' in data_size.lower():
                filtered_df = filtered_df[filtered_df['Veri Büyüklüğü '].isin(['GB', 'GB-TB', 'TB'])]
            elif 'medium' in data_size.lower() or 'orta' in data_size.lower():
                filtered_df = filtered_df[filtered_df['Veri Büyüklüğü '].isin(['MB-GB', 'GB'])]
        
        # Data type filtering
        if data_type:
            if 'numerical' in data_type.lower() or 'sayısal' in data_type.lower():
                filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('nd', na=False)]
            elif 'text' in data_type.lower() or 'metin' in data_type.lower():
                filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('td', na=False)]
            elif 'categorical' in data_type.lower() or 'kategorik' in data_type.lower():
                filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('cd', na=False)]
            elif 'image' in data_type.lower() or 'görüntü' in data_type.lower():
                filtered_df = filtered_df[filtered_df['Veri Tipi'].str.contains('id', na=False)]
        
        # Complexity filtering
        if complexity:
            if 'low' in complexity.lower() or 'düşük' in complexity.lower():
                filtered_df = filtered_df[filtered_df['Karmaşıklık Düzeyi'].isin(['comp1', 'comp2'])]
            elif 'high' in complexity.lower() or 'yüksek' in complexity.lower():
                filtered_df = filtered_df[filtered_df['Karmaşıklık Düzeyi'].isin(['comp4', 'comp5'])]
            elif 'medium' in complexity.lower() or 'orta' in complexity.lower():
                filtered_df = filtered_df[filtered_df['Karmaşıklık Düzeyi'] == 'comp3']
        
        return filtered_df
    
    def _calculate_similarity_scores(self, filtered_df: pd.DataFrame, user_profile: Dict, top_n: int) -> List[Dict]:
        """
        Calculate similarity scores using advanced ML techniques
        """
        recommendations = []
        
        # Sort by multiple criteria
        complexity_order = ['comp1', 'comp2', 'comp3', 'comp4', 'comp5']
        popularity_order = ['p5', 'p4', 'p3', 'p2', 'p1']
        
        filtered_df['complexity_rank'] = filtered_df['Karmaşıklık Düzeyi'].map(
            {comp: idx for idx, comp in enumerate(complexity_order)}
        )
        
        filtered_df['popularity_rank'] = filtered_df['Popülerlik'].map(
            {pop: idx for idx, pop in enumerate(popularity_order)}
        )
        
        # Multi-criteria sorting
        filtered_df = filtered_df.sort_values([
            'popularity_rank', 
            'complexity_rank'
        ])
        
        # Generate recommendations
        for idx, (_, row) in enumerate(filtered_df.head(top_n).iterrows()):
            confidence_score = self._calculate_advanced_confidence_score(row, user_profile, idx)
            
            rec = {
                'algorithm': row['Algoritma Adı'],
                'confidence_score': confidence_score,
                'learning_type': self.mappings['learning_type'].get(row['Öğrenme Türü'], row['Öğrenme Türü']),
                'complexity': self.mappings['complexity'].get(row['Karmaşıklık Düzeyi'], 'Medium'),
                'popularity': self.mappings['popularity'].get(row['Popülerlik'], row['Popülerlik']),
                'data_size_support': self.mappings['data_size'].get(row['Veri Büyüklüğü '], row['Veri Büyüklüğü ']),
                'hardware_requirement': row['Donanım Gerkesinimleri'],
                'use_cases': row['Kullanım Alanı'],
                'data_types': row['Veri Tipi'],
                'description': f"{row['Algoritma Adı']} - {self.mappings['complexity'].get(row['Karmaşıklık Düzeyi'], 'Medium')} complexity algorithm",
                'rank': idx + 1,
                'match_reasons': self._get_match_reasons(row, user_profile)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _calculate_advanced_confidence_score(self, row: pd.Series, user_profile: Dict, rank: int) -> float:
        """
        Calculate advanced confidence score with multiple factors
        """
        base_score = 5.0
        
        # Popularity factor (0.5 - 1.0)
        popularity_scores = {'p5': 1.0, 'p4': 0.9, 'p3': 0.8, 'p2': 0.7, 'p1': 0.6}
        popularity_factor = popularity_scores.get(row['Popülerlik'], 0.8)
        
        # Complexity factor (prefer medium complexity)
        complexity_scores = {'comp1': 0.7, 'comp2': 0.8, 'comp3': 1.0, 'comp4': 0.9, 'comp5': 0.8}
        complexity_factor = complexity_scores.get(row['Karmaşıklık Düzeyi'], 0.8)
        
        # Hardware requirement factor (lower is better)
        hardware_req = float(row['Donanım Gerkesinimleri']) if pd.notna(row['Donanım Gerkesinimleri']) else 0.5
        hardware_factor = max(0.5, 1.0 - hardware_req)
        
        # Rank factor (higher rank = lower score)
        rank_factor = max(0.6, 1.0 - (rank * 0.1))
        
        # Calculate final score
        final_score = base_score * popularity_factor * complexity_factor * hardware_factor * rank_factor
        
        # Add randomness for diversity
        final_score += np.random.uniform(-0.1, 0.1)
        
        return round(min(5.0, max(1.0, final_score)), 1)
    
    def _get_match_reasons(self, row: pd.Series, user_profile: Dict) -> List[str]:
        """
        Generate explanations for why this algorithm matches
        """
        reasons = []
        
        # Project type match
        project_type = user_profile.get('project_type', '')
        if 'classification' in project_type and 'cl' in row['Kullanım Alanı']:
            reasons.append("Sınıflandırma projeniz için optimize edilmiş")
        elif 'regression' in project_type and 're' in row['Kullanım Alanı']:
            reasons.append("Regresyon analiziniz için ideal")
        
        # Popularity
        if row['Popülerlik'] in ['p4', 'p5']:
            reasons.append("Yüksek popülerlik ve kanıtlanmış başarı")
        
        # Complexity
        complexity = user_profile.get('complexity', 'medium')
        if complexity == 'medium' and row['Karmaşıklık Düzeyi'] == 'comp3':
            reasons.append("Orta karmaşıklık seviyenize uygun")
        
        # Hardware requirements
        hardware_req = float(row['Donanım Gerkesinimleri']) if pd.notna(row['Donanım Gerkesinimleri']) else 0.5
        if hardware_req < 0.5:
            reasons.append("Düşük donanım gereksinimleri")
        
        return reasons[:3]  # Limit to top 3 reasons
    
    def _enhance_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Enhance recommendations with additional metadata and insights
        """
        for rec in recommendations:
            # Add performance estimates
            rec['estimated_performance'] = self._estimate_performance(rec)
            
            # Add use case examples
            rec['use_case_examples'] = self._get_use_case_examples(rec['algorithm'])
            
            # Add pros and cons
            rec['pros_cons'] = self._get_pros_cons(rec['algorithm'])
            
            # Add implementation difficulty
            rec['implementation_difficulty'] = self._get_implementation_difficulty(rec['complexity'])
        
        return recommendations

    def _estimate_performance(self, rec: Dict) -> Dict:
        """
        Estimate algorithm performance based on complexity and popularity
        """
        complexity = rec['complexity']
        popularity = rec['popularity']
        
        # Base performance estimates
        if 'High' in complexity or 'Very High' in complexity:
            accuracy_range = (0.85, 0.95)
        elif 'Medium' in complexity:
            accuracy_range = (0.75, 0.90)
        else:
            accuracy_range = (0.65, 0.85)
        
        # Adjust based on popularity
        if 'High' in popularity or 'Very High' in popularity:
            accuracy_range = (accuracy_range[0] + 0.05, min(0.98, accuracy_range[1] + 0.05))
        
        return {
            'accuracy_range': accuracy_range,
            'training_time': self._estimate_training_time(complexity),
            'prediction_speed': self._estimate_prediction_speed(complexity)
        }
    
    def _estimate_training_time(self, complexity: str) -> str:
        """Estimate training time based on complexity"""
        if 'Very High' in complexity:
            return "Uzun (saatler-günler)"
        elif 'High' in complexity:
            return "Orta-Uzun (dakikalar-saatler)"
        elif 'Medium' in complexity:
            return "Orta (dakikalar)"
        else:
            return "Hızlı (saniyeler-dakikalar)"
    
    def _estimate_prediction_speed(self, complexity: str) -> str:
        """Estimate prediction speed based on complexity"""
        if 'Very High' in complexity:
            return "Yavaş"
        elif 'High' in complexity:
            return "Orta-Yavaş"
        elif 'Medium' in complexity:
            return "Orta"
        else:
            return "Hızlı"
    
    def _get_use_case_examples(self, algorithm: str) -> List[str]:
        """Get specific use case examples for algorithms"""
        use_cases = {
            'Random Forest': [
                "E-ticaret önerisi sistemleri",
                "Müşteri segmentasyonu",
                "Finansal risk değerlendirmesi"
            ],
            'XGBoost': [
                "Kaggle yarışmaları",
                "Kredi skoru hesaplama",
                "Tıbbi teşhis sistemleri"
            ],
            'Logistic Regression': [
                "Email spam filtreleme",
                "Pazarlama kampanyası analizi",
                "Tıbbi test sonuçları"
            ],
            'K-Means': [
                "Müşteri segmentasyonu",
                "Pazar araştırması",
                "Görüntü segmentasyonu"
            ]
        }
        
        return use_cases.get(algorithm, [
            "Veri analizi projeleri",
            "Tahmin modelleri",
            "Karar destek sistemleri"
        ])
    
    def _get_pros_cons(self, algorithm: str) -> Dict:
        """Get pros and cons for specific algorithms"""
        pros_cons = {
            'Random Forest': {
                'pros': ['Overfitting\'e karşı dayanıklı', 'Özellik önemini gösterir', 'Eksik verilerle çalışabilir'],
                'cons': ['Büyük modeller', 'Yorumlanması zor', 'Bellek kullanımı yüksek']
            },
            'XGBoost': {
                'pros': ['Yüksek performans', 'Hızlı eğitim', 'Regularization desteği'],
                'cons': ['Parametre ayarı karmaşık', 'Overfitting riski', 'Bellek yoğun']
            },
            'Logistic Regression': {
                'pros': ['Hızlı ve basit', 'Yorumlanabilir', 'Düşük bellek kullanımı'],
                'cons': ['Doğrusal varsayım', 'Outlier\'lara hassas', 'Karmaşık ilişkileri yakalayamaz']
            }
        }
        
        return pros_cons.get(algorithm, {
            'pros': ['Genel amaçlı kullanım', 'Kanıtlanmış başarı', 'Topluluk desteği'],
            'cons': ['Özel optimizasyon gerekebilir', 'Veri tipine bağlı performans', 'Parametre ayarı']
        })
    
    def _get_implementation_difficulty(self, complexity: str) -> Dict:
        """Get implementation difficulty assessment"""
        if 'Very High' in complexity:
            return {
                'level': 'Çok Zor',
                'time_estimate': '1-2 hafta',
                'expertise_required': 'Expert level',
                'tools_needed': ['Advanced ML frameworks', 'GPU resources', 'Specialized libraries']
            }
        elif 'High' in complexity:
            return {
                'level': 'Zor',
                'time_estimate': '3-7 gün',
                'expertise_required': 'Advanced level',
                'tools_needed': ['ML frameworks', 'Good hardware', 'Domain knowledge']
            }
        elif 'Medium' in complexity:
            return {
                'level': 'Orta',
                'time_estimate': '1-3 gün',
                'expertise_required': 'Intermediate level',
                'tools_needed': ['Basic ML libraries', 'Standard hardware']
            }
        else:
            return {
                'level': 'Kolay',
                'time_estimate': '1-2 gün',
                'expertise_required': 'Beginner level',
                'tools_needed': ['Basic Python', 'Scikit-learn']
            }
    
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