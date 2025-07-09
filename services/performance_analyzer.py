import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
import io
import base64
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Advanced Performance Analysis Service for ML Algorithms
    """
    
    def __init__(self):
        self.analysis_history = []
        self.benchmark_results = {}
        
    def analyze_algorithm_performance(self, 
                                    algorithm_name: str, 
                                    project_type: str, 
                                    data_characteristics: Dict) -> Dict:
        """
        Comprehensive algorithm performance analysis
        """
        try:
            analysis = {
                'algorithm': algorithm_name,
                'project_type': project_type,
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': self._calculate_performance_metrics(algorithm_name, project_type),
                'scalability_analysis': self._analyze_scalability(algorithm_name, data_characteristics),
                'resource_requirements': self._analyze_resource_requirements(algorithm_name),
                'comparison_with_alternatives': self._compare_with_alternatives(algorithm_name, project_type),
                'optimization_suggestions': self._get_optimization_suggestions(algorithm_name),
                'deployment_considerations': self._get_deployment_considerations(algorithm_name),
                'visualization_data': self._generate_visualization_data(algorithm_name, project_type)
            }
            
            # Store in history
            self.analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return self._get_fallback_analysis(algorithm_name)
    
    def _calculate_performance_metrics(self, algorithm_name: str, project_type: str) -> Dict:
        """Calculate expected performance metrics"""
        
        # Performance benchmarks based on algorithm characteristics
        performance_data = {
            'Random Forest': {
                'classification': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.85, 'f1': 0.84},
                'regression': {'r2': 0.78, 'mse': 0.15, 'mae': 0.12}
            },
            'XGBoost': {
                'classification': {'accuracy': 0.88, 'precision': 0.87, 'recall': 0.88, 'f1': 0.87},
                'regression': {'r2': 0.82, 'mse': 0.12, 'mae': 0.10}
            },
            'Logistic Regression': {
                'classification': {'accuracy': 0.78, 'precision': 0.77, 'recall': 0.78, 'f1': 0.77},
                'regression': {'r2': 0.65, 'mse': 0.25, 'mae': 0.18}
            },
            'SVM': {
                'classification': {'accuracy': 0.82, 'precision': 0.81, 'recall': 0.82, 'f1': 0.81},
                'regression': {'r2': 0.75, 'mse': 0.18, 'mae': 0.14}
            },
            'K-Means': {
                'clustering': {'silhouette': 0.65, 'inertia': 0.35, 'adjusted_rand': 0.58}
            }
        }
        
        base_metrics = performance_data.get(algorithm_name, {}).get(project_type, {})
        
        if not base_metrics:
            # Default metrics for unknown algorithms
            if project_type == 'classification':
                base_metrics = {'accuracy': 0.75, 'precision': 0.74, 'recall': 0.75, 'f1': 0.74}
            elif project_type == 'regression':
                base_metrics = {'r2': 0.70, 'mse': 0.20, 'mae': 0.15}
            else:
                base_metrics = {'performance_score': 0.72}
        
        # Add confidence intervals
        enhanced_metrics = {}
        for metric, value in base_metrics.items():
            enhanced_metrics[metric] = {
                'expected': value,
                'confidence_interval': (value - 0.05, value + 0.05),
                'best_case': min(1.0, value + 0.10),
                'worst_case': max(0.0, value - 0.10)
            }
        
        return enhanced_metrics
    
    def _analyze_scalability(self, algorithm_name: str, data_characteristics: Dict) -> Dict:
        """Analyze algorithm scalability"""
        
        scalability_profiles = {
            'Random Forest': {
                'data_size_scaling': 'Good',
                'feature_scaling': 'Excellent',
                'time_complexity': 'O(n * m * log(n))',
                'space_complexity': 'O(n * m)',
                'parallel_processing': 'Excellent',
                'memory_efficiency': 'Good'
            },
            'XGBoost': {
                'data_size_scaling': 'Excellent',
                'feature_scaling': 'Good',
                'time_complexity': 'O(n * m * d)',
                'space_complexity': 'O(n * m)',
                'parallel_processing': 'Excellent',
                'memory_efficiency': 'Good'
            },
            'Logistic Regression': {
                'data_size_scaling': 'Excellent',
                'feature_scaling': 'Good',
                'time_complexity': 'O(n * m)',
                'space_complexity': 'O(m)',
                'parallel_processing': 'Good',
                'memory_efficiency': 'Excellent'
            },
            'SVM': {
                'data_size_scaling': 'Poor',
                'feature_scaling': 'Good',
                'time_complexity': 'O(n²) to O(n³)',
                'space_complexity': 'O(n²)',
                'parallel_processing': 'Limited',
                'memory_efficiency': 'Poor'
            }
        }
        
        profile = scalability_profiles.get(algorithm_name, {
            'data_size_scaling': 'Average',
            'feature_scaling': 'Average',
            'time_complexity': 'O(n * m)',
            'space_complexity': 'O(n * m)',
            'parallel_processing': 'Limited',
            'memory_efficiency': 'Average'
        })
        
        # Adjust based on data characteristics
        data_size = data_characteristics.get('size', 'medium')
        if data_size == 'large' and profile['data_size_scaling'] == 'Poor':
            profile['recommendation'] = 'Consider alternative algorithms for large datasets'
        elif data_size == 'small' and profile['data_size_scaling'] == 'Excellent':
            profile['recommendation'] = 'Might be overkill for small datasets'
        else:
            profile['recommendation'] = 'Good fit for your data size'
        
        return profile
    
    def _analyze_resource_requirements(self, algorithm_name: str) -> Dict:
        """Analyze computational resource requirements"""
        
        resource_profiles = {
            'Random Forest': {
                'cpu_usage': 'High',
                'memory_usage': 'High',
                'gpu_requirement': 'Optional',
                'disk_space': 'Medium',
                'training_time': 'Medium',
                'prediction_time': 'Fast',
                'recommended_hardware': {
                    'cpu_cores': '4-8',
                    'ram': '8-16 GB',
                    'storage': '10-50 GB'
                }
            },
            'XGBoost': {
                'cpu_usage': 'High',
                'memory_usage': 'Medium',
                'gpu_requirement': 'Beneficial',
                'disk_space': 'Low',
                'training_time': 'Fast',
                'prediction_time': 'Very Fast',
                'recommended_hardware': {
                    'cpu_cores': '4-16',
                    'ram': '4-8 GB',
                    'storage': '5-20 GB'
                }
            },
            'Logistic Regression': {
                'cpu_usage': 'Low',
                'memory_usage': 'Low',
                'gpu_requirement': 'Not needed',
                'disk_space': 'Very Low',
                'training_time': 'Very Fast',
                'prediction_time': 'Very Fast',
                'recommended_hardware': {
                    'cpu_cores': '2-4',
                    'ram': '2-4 GB',
                    'storage': '1-5 GB'
                }
            },
            'Deep Learning': {
                'cpu_usage': 'Very High',
                'memory_usage': 'Very High',
                'gpu_requirement': 'Essential',
                'disk_space': 'High',
                'training_time': 'Very Slow',
                'prediction_time': 'Medium',
                'recommended_hardware': {
                    'cpu_cores': '8-32',
                    'ram': '16-64 GB',
                    'storage': '100-500 GB',
                    'gpu': 'High-end GPU recommended'
                }
            }
        }
        
        return resource_profiles.get(algorithm_name, {
            'cpu_usage': 'Medium',
            'memory_usage': 'Medium',
            'gpu_requirement': 'Optional',
            'disk_space': 'Medium',
            'training_time': 'Medium',
            'prediction_time': 'Medium',
            'recommended_hardware': {
                'cpu_cores': '4',
                'ram': '8 GB',
                'storage': '20 GB'
            }
        })
    
    def _compare_with_alternatives(self, algorithm_name: str, project_type: str) -> Dict:
        """Compare with alternative algorithms"""
        
        alternatives = {
            'Random Forest': {
                'classification': [
                    {'name': 'XGBoost', 'performance': '+5%', 'speed': '+20%', 'complexity': '+15%'},
                    {'name': 'SVM', 'performance': '-3%', 'speed': '-40%', 'complexity': '-10%'},
                    {'name': 'Logistic Regression', 'performance': '-8%', 'speed': '+80%', 'complexity': '-50%'}
                ],
                'regression': [
                    {'name': 'XGBoost', 'performance': '+7%', 'speed': '+15%', 'complexity': '+20%'},
                    {'name': 'Linear Regression', 'performance': '-15%', 'speed': '+90%', 'complexity': '-60%'},
                    {'name': 'SVR', 'performance': '-5%', 'speed': '-30%', 'complexity': '+10%'}
                ]
            },
            'XGBoost': {
                'classification': [
                    {'name': 'Random Forest', 'performance': '-5%', 'speed': '-20%', 'complexity': '-15%'},
                    {'name': 'LightGBM', 'performance': '-2%', 'speed': '+30%', 'complexity': '0%'},
                    {'name': 'CatBoost', 'performance': '+2%', 'speed': '-10%', 'complexity': '-5%'}
                ]
            }
        }
        
        return alternatives.get(algorithm_name, {}).get(project_type, [])
    
    def _get_optimization_suggestions(self, algorithm_name: str) -> List[Dict]:
        """Get optimization suggestions for the algorithm"""
        
        optimizations = {
            'Random Forest': [
                {
                    'category': 'Hyperparameter Tuning',
                    'suggestion': 'n_estimators optimize etme',
                    'impact': 'High',
                    'difficulty': 'Easy',
                    'details': 'GridSearchCV ile 100-500 arasında test edin'
                },
                {
                    'category': 'Feature Engineering',
                    'suggestion': 'Feature importance analizi',
                    'impact': 'Medium',
                    'difficulty': 'Easy',
                    'details': 'Düşük önemli özellikleri kaldırın'
                },
                {
                    'category': 'Performance',
                    'suggestion': 'Paralel işleme',
                    'impact': 'High',
                    'difficulty': 'Easy',
                    'details': 'n_jobs=-1 parametresini kullanın'
                }
            ],
            'XGBoost': [
                {
                    'category': 'Hyperparameter Tuning',
                    'suggestion': 'Learning rate ve max_depth optimize etme',
                    'impact': 'Very High',
                    'difficulty': 'Medium',
                    'details': 'Bayesian optimization kullanın'
                },
                {
                    'category': 'Regularization',
                    'suggestion': 'L1/L2 regularization',
                    'impact': 'High',
                    'difficulty': 'Medium',
                    'details': 'alpha ve lambda parametrelerini ayarlayın'
                },
                {
                    'category': 'Early Stopping',
                    'suggestion': 'Erken durdurma',
                    'impact': 'Medium',
                    'difficulty': 'Easy',
                    'details': 'Validation set ile overfitting önleyin'
                }
            ],
            'Logistic Regression': [
                {
                    'category': 'Regularization',
                    'suggestion': 'Ridge/Lasso regularization',
                    'impact': 'Medium',
                    'difficulty': 'Easy',
                    'details': 'C parametresini optimize edin'
                },
                {
                    'category': 'Feature Scaling',
                    'suggestion': 'Standardization',
                    'impact': 'High',
                    'difficulty': 'Easy',
                    'details': 'StandardScaler kullanın'
                },
                {
                    'category': 'Solver Selection',
                    'suggestion': 'Uygun solver seçimi',
                    'impact': 'Medium',
                    'difficulty': 'Easy',
                    'details': 'Veri boyutuna göre liblinear/lbfgs seçin'
                }
            ]
        }
        
        return optimizations.get(algorithm_name, [
            {
                'category': 'General',
                'suggestion': 'Hyperparameter tuning',
                'impact': 'High',
                'difficulty': 'Medium',
                'details': 'Grid search veya random search kullanın'
            }
        ])
    
    def _get_deployment_considerations(self, algorithm_name: str) -> Dict:
        """Get deployment considerations"""
        
        deployment_info = {
            'Random Forest': {
                'model_size': 'Large',
                'inference_speed': 'Fast',
                'scalability': 'Good',
                'maintenance': 'Low',
                'monitoring_needs': [
                    'Feature drift detection',
                    'Performance degradation',
                    'Resource usage monitoring'
                ],
                'deployment_options': [
                    'Batch processing',
                    'Real-time API',
                    'Edge deployment (with optimization)'
                ],
                'considerations': [
                    'Model boyutu büyük olabilir',
                    'Paralel işleme avantajı',
                    'Feature importance takibi önemli'
                ]
            },
            'XGBoost': {
                'model_size': 'Medium',
                'inference_speed': 'Very Fast',
                'scalability': 'Excellent',
                'maintenance': 'Medium',
                'monitoring_needs': [
                    'Model performance tracking',
                    'Data drift detection',
                    'Hyperparameter sensitivity'
                ],
                'deployment_options': [
                    'Real-time API',
                    'Batch processing',
                    'Mobile deployment',
                    'Edge computing'
                ],
                'considerations': [
                    'Hızlı inference',
                    'Düşük latency',
                    'Model versioning önemli'
                ]
            },
            'Logistic Regression': {
                'model_size': 'Very Small',
                'inference_speed': 'Very Fast',
                'scalability': 'Excellent',
                'maintenance': 'Very Low',
                'monitoring_needs': [
                    'Coefficient stability',
                    'Feature scaling consistency'
                ],
                'deployment_options': [
                    'Real-time API',
                    'Mobile deployment',
                    'Edge computing',
                    'Embedded systems'
                ],
                'considerations': [
                    'Çok hafif model',
                    'Hızlı deployment',
                    'Düşük kaynak kullanımı'
                ]
            }
        }
        
        return deployment_info.get(algorithm_name, {
            'model_size': 'Medium',
            'inference_speed': 'Medium',
            'scalability': 'Good',
            'maintenance': 'Medium',
            'monitoring_needs': ['Performance monitoring'],
            'deployment_options': ['API deployment'],
            'considerations': ['Standard deployment yaklaşımı']
        })
    
    def _generate_visualization_data(self, algorithm_name: str, project_type: str) -> Dict:
        """Generate data for performance visualizations"""
        
        # Generate synthetic performance data for visualization
        np.random.seed(42)
        
        # Performance over time
        time_points = list(range(1, 11))
        performance_trend = np.random.normal(0.8, 0.05, 10)
        performance_trend = np.clip(performance_trend, 0.6, 0.95)
        
        # Resource usage over data size
        data_sizes = [100, 500, 1000, 5000, 10000, 50000]
        cpu_usage = [x * 0.001 + np.random.normal(0, 0.1) for x in data_sizes]
        memory_usage = [x * 0.002 + np.random.normal(0, 0.2) for x in data_sizes]
        
        # Comparison with alternatives
        algorithms = ['Current', 'Alternative 1', 'Alternative 2', 'Alternative 3']
        metrics = ['Accuracy', 'Speed', 'Memory', 'Complexity']
        comparison_data = np.random.uniform(0.6, 0.9, (4, 4))
        
        return {
            'performance_trend': {
                'x': time_points,
                'y': performance_trend.tolist(),
                'title': f'{algorithm_name} Performance Over Time',
                'xlabel': 'Training Epochs',
                'ylabel': 'Performance Score'
            },
            'resource_usage': {
                'data_sizes': data_sizes,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'title': f'{algorithm_name} Resource Usage vs Data Size'
            },
            'algorithm_comparison': {
                'algorithms': algorithms,
                'metrics': metrics,
                'data': comparison_data.tolist(),
                'title': f'{algorithm_name} vs Alternatives'
            }
        }
    
    def _get_fallback_analysis(self, algorithm_name: str) -> Dict:
        """Fallback analysis when main analysis fails"""
        return {
            'algorithm': algorithm_name,
            'timestamp': datetime.now().isoformat(),
            'status': 'fallback',
            'message': 'Detaylı analiz şu anda mevcut değil',
            'basic_info': {
                'recommendation': f'{algorithm_name} genel olarak iyi bir seçim',
                'next_steps': [
                    'Veri setinizle test edin',
                    'Hyperparameter tuning yapın',
                    'Cross-validation kullanın'
                ]
            }
        }
    
    def get_analysis_history(self) -> List[Dict]:
        """Get analysis history"""
        return self.analysis_history
    
    def export_analysis_report(self, analysis: Dict) -> str:
        """Export analysis as formatted report"""
        report = f"""
# {analysis['algorithm']} Performance Analysis Report

## Overview
- Algorithm: {analysis['algorithm']}
- Project Type: {analysis['project_type']}
- Analysis Date: {analysis['timestamp']}

## Performance Metrics
{json.dumps(analysis.get('performance_metrics', {}), indent=2)}

## Scalability Analysis
{json.dumps(analysis.get('scalability_analysis', {}), indent=2)}

## Resource Requirements
{json.dumps(analysis.get('resource_requirements', {}), indent=2)}

## Optimization Suggestions
{json.dumps(analysis.get('optimization_suggestions', []), indent=2)}

## Deployment Considerations
{json.dumps(analysis.get('deployment_considerations', {}), indent=2)}
        """
        
        return report 