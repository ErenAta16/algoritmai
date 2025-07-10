"""
Conversation Management Service - Separated from OpenAIService for better architecture
"""

import time
import logging
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation memory, context, and user profiling
    """
    
    def __init__(self):
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
    
    def update_conversation_memory(self, user_message: str, conversation_history: Optional[List[Dict]] = None):
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
    
    def _track_algorithm_mentions(self, user_message: str):
        """Track mentioned algorithms in conversation"""
        text_lower = user_message.lower()
        
        # Algorithm names to track - expanded list
        algorithms = {
            'xgboost': ['xgboost', 'xgb'],
            'random forest': ['random forest', 'rf'],
            'svm': ['svm', 'support vector'],
            'neural network': ['neural network', 'nn', 'deep learning'],
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
            'ada boost': ['ada boost', 'adaboost']
        }
        
        for algorithm, variants in algorithms.items():
            for variant in variants:
                if variant in text_lower:
                    if algorithm not in self.conversation_context['discussed_algorithms']:
                        self.conversation_context['discussed_algorithms'].append(algorithm)
                        logger.info(f"📝 Algorithm mentioned: {algorithm}")
                    break
    
    def _track_user_preferences(self, user_message: str):
        """Track user preferences and feedback"""
        text_lower = user_message.lower()
        
        # Positive feedback
        if any(word in text_lower for word in ['güzel', 'iyi', 'beğendim', 'teşekkür', 'mükemmel', 'harika']):
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
    
    def analyze_user_profile(self, user_message: str):
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
    
    def store_response_for_diversity(self, user_message: str, response: str):
        """Store response for future diversity checking"""
        # Keep only last 20 responses to prevent memory bloat
        if len(self.response_cache) >= 20:
            # Remove oldest entry
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[user_message] = response
    
    def find_similar_responses(self, user_message: str) -> List[str]:
        """Find similar previous responses to avoid repetition"""
        similar_responses = []
        
        # Simple similarity check based on common words
        user_words = set(user_message.lower().split())
        
        for prev_message, prev_response in self.response_cache.items():
            prev_words = set(prev_message.lower().split())
            
            # Calculate word overlap
            overlap = len(user_words.intersection(prev_words))
            if overlap > 2:  # If more than 2 words overlap
                similar_responses.append(prev_response)
        
        return similar_responses
    
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for AI prompts"""
        context_parts = []
        
        if self.conversation_memory:
            context_parts.append(f"Conversation length: {len(self.conversation_memory)} messages")
        
        if self.conversation_context['discussed_algorithms']:
            algorithms_str = ", ".join(self.conversation_context['discussed_algorithms'][-5:])
            context_parts.append(f"Discussed algorithms: {algorithms_str}")
        
        if self.conversation_context['last_recommendations']:
            context_parts.append(f"Last recommendations count: {len(self.conversation_context['last_recommendations'])}")
        
        user_feedback_summary = self._summarize_user_feedback()
        if user_feedback_summary:
            context_parts.append(f"User feedback: {user_feedback_summary}")
        
        return " | ".join(context_parts)
    
    def _summarize_user_feedback(self) -> str:
        """Summarize recent user feedback"""
        if not self.conversation_context['user_feedback']:
            return ""
        
        recent_feedback = self.conversation_context['user_feedback'][-3:]  # Last 3 feedback items
        
        feedback_types = [f['type'] for f in recent_feedback]
        
        if 'positive' in feedback_types:
            return "Generally positive"
        elif 'negative' in feedback_types:
            return "Some concerns expressed"
        elif 'question' in feedback_types:
            return "Asking detailed questions"
        else:
            return "Neutral engagement"
    
    def get_conversation_stage(self, context: Dict) -> str:
        """Determine what stage of conversation we're in"""
        if not context.get('project_type'):
            return 'initial_consultation'
        elif context.get('project_type') and not context.get('data_size'):
            return 'gathering_requirements'
        elif all(context.get(field) for field in ['project_type', 'data_size', 'data_type']):
            return 'ready_for_recommendations'
        else:
            return 'ongoing_consultation' 