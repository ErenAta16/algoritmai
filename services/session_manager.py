"""
Advanced Session Management System for AI Algorithm Consultant
Handles context persistence, conversation state, and intelligent merging
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
import pickle
import os

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Advanced session management with persistent context and intelligent merging
    """
    
    def __init__(self):
        self.sessions = {}  # session_id -> session_data
        self.session_timeout = 3600  # 1 hour
        self.max_sessions = 1000
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Session data structure
        self.session_template = {
            'created_at': None,
            'last_activity': None,
            'conversation_memory': [],
            'project_context': {
                'project_type': None,
                'data_size': None,
                'data_type': None,
                'class_count': None,
                'use_case': None,
                'constraints': [],
                'mentioned_algorithms': [],
                'conversation_stage': 'initial'
            },
            'user_profile': {
                'experience_level': 'unknown',
                'preferred_style': 'unknown',
                'project_domain': 'unknown',
                'technical_comfort': 'unknown'
            },
            'conversation_context': {
                'last_recommendations': [],
                'user_selections': [],
                'discussed_algorithms': [],
                'user_feedback': []
            },
            'response_diversity': {
                'response_cache': {},
                'response_variations': {},
                'conversation_turn': 0
            },
            'metadata': {
                'total_messages': 0,
                'successful_recommendations': 0,
                'user_satisfaction': 'unknown'
            }
        }
    
    def get_or_create_session(self, session_id: str = None) -> str:
        """Get existing session or create new one"""
        if not session_id:
            session_id = self._generate_session_id()
        
        # Clean up old sessions periodically
        if time.time() - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_sessions()
        
        # Create new session if doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()
            logger.info(f"✅ Created new session: {session_id}")
        
        # Update last activity
        self.sessions[session_id]['last_activity'] = time.time()
        
        return session_id
    
    def get_session_context(self, session_id: str) -> Dict:
        """Get complete session context"""
        if session_id not in self.sessions:
            return self._create_new_session()
        
        return self.sessions[session_id]
    
    def update_session_context(self, session_id: str, context_updates: Dict):
        """Update session context with intelligent merging"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()
        
        session = self.sessions[session_id]
        
        # Merge project context intelligently
        if 'project_context' in context_updates:
            self._merge_project_context(session['project_context'], context_updates['project_context'])
        
        # Merge conversation context
        if 'conversation_context' in context_updates:
            self._merge_conversation_context(session['conversation_context'], context_updates['conversation_context'])
        
        # Update user profile
        if 'user_profile' in context_updates:
            self._merge_user_profile(session['user_profile'], context_updates['user_profile'])
        
        # Update response diversity tracking
        if 'response_diversity' in context_updates:
            self._merge_response_diversity(session['response_diversity'], context_updates['response_diversity'])
        
        # Update metadata
        session['metadata']['total_messages'] += 1
        session['last_activity'] = time.time()
        
        logger.info(f"📊 Updated session context: {session_id}")
    
    def merge_conversation_history(self, session_id: str, frontend_history: List[Dict]) -> List[Dict]:
        """Intelligently merge frontend history with backend context"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()
        
        session = self.sessions[session_id]
        backend_memory = session['conversation_memory']
        
        # If no backend memory, use frontend history
        if not backend_memory:
            session['conversation_memory'] = frontend_history[-20:]  # Keep last 20
            return session['conversation_memory']
        
        # Merge intelligently
        merged_memory = self._intelligent_memory_merge(backend_memory, frontend_history)
        session['conversation_memory'] = merged_memory
        
        return merged_memory
    
    def extract_persistent_context(self, session_id: str, current_message: str, frontend_history: List[Dict]) -> Dict:
        """Extract context with persistence across requests"""
        if session_id not in self.sessions:
            self.sessions[session_id] = self._create_new_session()
        
        session = self.sessions[session_id]
        
        # Start with existing context
        context = session['project_context'].copy()
        
        # Add conversation metadata
        context['conversation_turn'] = session['response_diversity']['conversation_turn']
        context['conversation_length'] = len(session['conversation_memory'])
        context['conversation_context'] = session['conversation_context']
        context['conversation_memory'] = session['conversation_memory']
        
        # Extract new information from current message and history
        new_context = self._extract_new_context_info(current_message, frontend_history)
        
        # Merge with existing context (preserve existing values)
        for key, value in new_context.items():
            if value is not None and (context.get(key) is None or context.get(key) == 'unknown'):
                context[key] = value
        
        # Update session with merged context
        session['project_context'].update(context)
        
        return context
    
    def track_algorithm_mentions(self, session_id: str, user_message: str, recommendations: List[Dict] = None):
        """Track algorithm mentions across conversation"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        text_lower = user_message.lower()
        
        # Algorithm keywords mapping
        algorithms = {
            'xgboost': ['xgboost', 'xgb', 'extreme gradient boosting'],
            'random forest': ['random forest', 'rf', 'random tree'],
            'svm': ['svm', 'support vector', 'support vector machine'],
            'neural network': ['neural network', 'neural', 'nn', 'deep learning', 'mlp'],
            'logistic regression': ['logistic regression', 'logistic', 'logit'],
            'naive bayes': ['naive bayes', 'nb', 'bayes'],
            'knn': ['knn', 'k-nearest', 'k nearest neighbor'],
            'decision tree': ['decision tree', 'dt', 'tree'],
            'linear regression': ['linear regression', 'linear model'],
            'k-means': ['kmeans', 'k-means', 'k means'],
            'dbscan': ['dbscan', 'density clustering'],
            'optics': ['optics'],
            'mean shift': ['mean shift'],
            'ensemble': ['ensemble', 'bagging', 'boosting'],
            'gradient boosting': ['gradient boosting', 'gbm', 'gradientboosting'],
            'ada boost': ['ada boost', 'adaboost', 'adaptive boosting'],
            'lightgbm': ['lightgbm', 'lgbm', 'light gbm'],
            'catboost': ['catboost', 'cat boost'],
            'prophet': ['prophet', 'facebook prophet'],
            'lstm': ['lstm', 'long short term memory'],
            'cnn': ['cnn', 'convolutional neural network'],
            'transformer': ['transformer', 'bert', 'gpt'],
            'pca': ['pca', 'principal component analysis'],
            'tsne': ['tsne', 't-sne', 'stochastic neighbor embedding']
        }
        
        # Track mentioned algorithms
        for algo_name, keywords in algorithms.items():
            if any(keyword in text_lower for keyword in keywords):
                if algo_name not in session['conversation_context']['discussed_algorithms']:
                    session['conversation_context']['discussed_algorithms'].append(algo_name)
                    logger.info(f"📝 Tracked algorithm mention: {algo_name}")
        
        # Track recommendations if provided
        if recommendations:
            for rec in recommendations:
                algo_name = rec.get('algorithm', '').lower()
                if algo_name and algo_name not in session['conversation_context']['discussed_algorithms']:
                    session['conversation_context']['discussed_algorithms'].append(algo_name)
            
            session['conversation_context']['last_recommendations'] = recommendations
            session['metadata']['successful_recommendations'] += 1
    
    def get_response_diversity_context(self, session_id: str, user_message: str) -> Dict:
        """Get context for response diversity"""
        if session_id not in self.sessions:
            return {'similar_responses': [], 'conversation_turn': 0}
        
        session = self.sessions[session_id]
        response_cache = session['response_diversity']['response_cache']
        
        # Find similar responses
        similar_responses = []
        user_words = set(user_message.lower().split())
        
        for cached_message, cached_response in response_cache.items():
            cached_words = set(cached_message.lower().split())
            
            if user_words and cached_words:
                intersection = user_words.intersection(cached_words)
                similarity = len(intersection) / len(user_words.union(cached_words))
                
                if similarity > 0.5:  # 50% similarity threshold
                    similar_responses.append(cached_response)
        
        return {
            'similar_responses': similar_responses,
            'conversation_turn': session['response_diversity']['conversation_turn'],
            'previous_responses': list(response_cache.values())[-3:]  # Last 3 responses
        }
    
    def store_response_for_diversity(self, session_id: str, user_message: str, response: str):
        """Store response for diversity tracking"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Store response
        session['response_diversity']['response_cache'][user_message] = response
        session['response_diversity']['conversation_turn'] += 1
        
        # Keep only last 50 responses for memory efficiency
        if len(session['response_diversity']['response_cache']) > 50:
            cache_items = list(session['response_diversity']['response_cache'].items())
            session['response_diversity']['response_cache'] = dict(cache_items[-50:])
    
    def _create_new_session(self) -> Dict:
        """Create new session with template"""
        session = self.session_template.copy()
        session['created_at'] = time.time()
        session['last_activity'] = time.time()
        
        # Deep copy nested dictionaries
        session['project_context'] = self.session_template['project_context'].copy()
        session['user_profile'] = self.session_template['user_profile'].copy()
        session['conversation_context'] = {
            'last_recommendations': [],
            'user_selections': [],
            'discussed_algorithms': [],
            'user_feedback': []
        }
        session['response_diversity'] = {
            'response_cache': {},
            'response_variations': {},
            'conversation_turn': 0
        }
        session['metadata'] = self.session_template['metadata'].copy()
        
        return session
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def _cleanup_old_sessions(self):
        """Clean up expired sessions"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, session_data in self.sessions.items():
            if current_time - session_data['last_activity'] > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"🧹 Cleaned up expired session: {session_id}")
        
        self.last_cleanup = current_time
        
        # Limit total sessions
        if len(self.sessions) > self.max_sessions:
            # Remove oldest sessions
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1]['last_activity']
            )
            
            sessions_to_remove = len(self.sessions) - self.max_sessions
            for i in range(sessions_to_remove):
                session_id = sorted_sessions[i][0]
                del self.sessions[session_id]
                logger.info(f"🧹 Removed old session due to limit: {session_id}")
    
    def _merge_project_context(self, existing: Dict, new: Dict):
        """Merge project context intelligently"""
        for key, value in new.items():
            if value is not None:
                if existing.get(key) is None or existing.get(key) == 'unknown':
                    existing[key] = value
                elif key == 'mentioned_algorithms':
                    # Merge algorithm lists
                    if isinstance(value, list):
                        for algo in value:
                            if algo not in existing[key]:
                                existing[key].append(algo)
                elif key == 'constraints':
                    # Merge constraint lists
                    if isinstance(value, list):
                        for constraint in value:
                            if constraint not in existing[key]:
                                existing[key].append(constraint)
    
    def _merge_conversation_context(self, existing: Dict, new: Dict):
        """Merge conversation context"""
        for key, value in new.items():
            if key in ['last_recommendations']:
                existing[key] = value  # Replace with latest
            elif key in ['discussed_algorithms', 'user_selections', 'user_feedback']:
                # Merge lists
                if isinstance(value, list):
                    for item in value:
                        if item not in existing[key]:
                            existing[key].append(item)
    
    def _merge_user_profile(self, existing: Dict, new: Dict):
        """Merge user profile"""
        for key, value in new.items():
            if value is not None and value != 'unknown':
                existing[key] = value
    
    def _merge_response_diversity(self, existing: Dict, new: Dict):
        """Merge response diversity data"""
        if 'response_cache' in new:
            existing['response_cache'].update(new['response_cache'])
        if 'conversation_turn' in new:
            existing['conversation_turn'] = max(existing['conversation_turn'], new['conversation_turn'])
    
    def _intelligent_memory_merge(self, backend_memory: List[Dict], frontend_history: List[Dict]) -> List[Dict]:
        """Intelligently merge backend memory with frontend history"""
        # Create a combined list and remove duplicates
        combined = []
        seen_contents = set()
        
        # Add backend memory first (it's more reliable)
        for msg in backend_memory:
            content = msg.get('content', '')
            if content and content not in seen_contents:
                combined.append(msg)
                seen_contents.add(content)
        
        # Add frontend history, avoiding duplicates
        for msg in frontend_history:
            content = msg.get('content', '')
            if content and content not in seen_contents:
                combined.append(msg)
                seen_contents.add(content)
        
        # Sort by timestamp if available, otherwise maintain order
        try:
            combined.sort(key=lambda x: x.get('timestamp', 0))
        except:
            pass
        
        # Keep only last 20 messages
        return combined[-20:]
    
    def _extract_new_context_info(self, current_message: str, frontend_history: List[Dict]) -> Dict:
        """Extract new context information from current message and history"""
        context = {}
        
        # Combine all conversation content
        full_conversation = ""
        if frontend_history:
            for msg in frontend_history[-10:]:  # Last 10 messages
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    full_conversation += f"{content}\n"
        
        full_conversation += current_message
        text_lower = full_conversation.lower()
        
        # Project type detection - more comprehensive
        if any(word in text_lower for word in ['sınıflandırma', 'siniflandirma', 'classification', 'kategorilere ayır', 'sınıflama', 'classify', 'kategori']):
            context['project_type'] = 'classification'
        elif any(word in text_lower for word in ['kümeleme', 'kumeleme', 'clustering', 'segmentasyon', 'gruplama', 'cluster', 'group']):
            context['project_type'] = 'clustering'
        elif any(word in text_lower for word in ['regresyon', 'regression', 'değer tahmin', 'fiyat tahmin', 'predict', 'tahmin']):
            context['project_type'] = 'regression'
        elif any(word in text_lower for word in ['anomali', 'outlier', 'anormal', 'dolandırıcılık', 'fraud']):
            context['project_type'] = 'anomaly_detection'
        elif any(word in text_lower for word in ['öneri', 'recommendation', 'tavsiye', 'recommend']):
            context['project_type'] = 'recommendation'
        elif any(word in text_lower for word in ['zaman serisi', 'time series', 'temporal', 'zaman']):
            context['project_type'] = 'time_series'
        
        # Data type detection
        if any(word in text_lower for word in ['sayısal', 'numerical', 'numeric', 'number', 'float', 'int']):
            context['data_type'] = 'numerical'
        elif any(word in text_lower for word in ['kategorik', 'categorical', 'category', 'string', 'text']):
            context['data_type'] = 'categorical'
        elif any(word in text_lower for word in ['metin', 'text', 'kelime', 'word', 'nlp', 'natural language']):
            context['data_type'] = 'text'
        elif any(word in text_lower for word in ['görüntü', 'image', 'resim', 'photo', 'cv', 'computer vision']):
            context['data_type'] = 'image'
        
        # Data size detection - more intelligent
        import re
        numbers = re.findall(r'\d+', text_lower)
        for num in numbers:
            num_val = int(num)
            if num_val < 1000:
                context['data_size'] = 'small'
                break
            elif num_val < 10000:
                context['data_size'] = 'medium'
                break
            else:
                context['data_size'] = 'large'
                break
        
        # Size indicators without numbers
        if 'data_size' not in context:
            if any(word in text_lower for word in ['küçük', 'small', 'az', 'little']):
                context['data_size'] = 'small'
            elif any(word in text_lower for word in ['büyük', 'large', 'çok', 'many', 'milyon']):
                context['data_size'] = 'large'
            elif any(word in text_lower for word in ['orta', 'medium', 'normal']):
                context['data_size'] = 'medium'
        
        # Default data size if not detected but project type is known
        if 'data_size' not in context and context.get('project_type'):
            context['data_size'] = 'medium'
        
        # Complexity preference detection
        if any(word in text_lower for word in ['basit', 'simple', 'kolay', 'easy', 'hızlı', 'fast']):
            context['complexity_preference'] = 'low'
        elif any(word in text_lower for word in ['karmaşık', 'complex', 'gelişmiş', 'advanced', 'sophisticated']):
            context['complexity_preference'] = 'high'
        elif any(word in text_lower for word in ['orta', 'medium', 'balanced']):
            context['complexity_preference'] = 'medium'
        
        return context

# Global session manager instance
session_manager = SessionManager() 