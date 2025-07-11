import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import '@fortawesome/fontawesome-free/css/all.min.css';

// Import components
import Navigation from './components/Navigation';
import Auth from './components/Auth';
import AlgorithmRecommendation from './components/AlgorithmRecommendation';
import UserProfile from './components/UserProfile';

// Styled Components - Professional-test.html'den dönüştürüldü
const AppContainer = styled.div`
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
  min-height: 100vh;
  color: #ffffff;
  line-height: 1.6;
  overflow-x: hidden;
`;

const Container = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
`;

const Header = styled.header`
  text-align: center;
  margin-bottom: 3rem;
`;

const Title = styled.h1`
  font-size: 3rem;
  font-weight: 800;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 1rem;
`;

const Subtitle = styled.p`
  font-size: 1.2rem;
  color: #94a3b8;
  margin-bottom: 2rem;
`;

const StatusBar = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem 1.5rem;
  margin-bottom: 2rem;
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
`;

const StatusIndicator = styled.div<{ $isOnline: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.$isOnline ? '#10b981' : '#ef4444'};
  animation: ${props => props.$isOnline ? 'pulse 2s infinite' : 'none'};
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
`;

const ChatContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  overflow: hidden;
`;

const ChatHeader = styled.div`
  background: rgba(255, 255, 255, 0.1);
  padding: 1.5rem;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const ChatTitle = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 1.5rem;
  overflow-y: auto;
  max-height: 500px;
`;

const Message = styled.div<{ $isUser: boolean }>`
  display: flex;
  margin-bottom: 1.5rem;
  justify-content: ${props => props.$isUser ? 'flex-end' : 'flex-start'};
`;

const MessageBubble = styled.div<{ $isUser: boolean }>`
  max-width: 70%;
  padding: 1rem 1.5rem;
  border-radius: 18px;
  background: ${props => props.$isUser 
    ? 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)' 
    : 'rgba(255, 255, 255, 0.1)'};
  border: 1px solid ${props => props.$isUser ? 'transparent' : 'rgba(255, 255, 255, 0.2)'};
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 50%;
    ${props => props.$isUser ? 'right: -8px' : 'left: -8px'};
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border: 8px solid transparent;
    border-${props => props.$isUser ? 'left' : 'right'}-color: ${props => props.$isUser 
      ? '#3b82f6' : 'rgba(255, 255, 255, 0.1)'};
  }
`;

const MessageText = styled.p`
  margin: 0;
  line-height: 1.5;
  white-space: pre-wrap;
`;

const InputContainer = styled.div`
  padding: 1.5rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.05);
`;

const InputForm = styled.form`
  display: flex;
  gap: 1rem;
  align-items: flex-end;
`;

const InputField = styled.textarea`
  flex: 1;
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  padding: 1rem;
  color: #ffffff;
  font-size: 1rem;
  resize: none;
  min-height: 50px;
  max-height: 120px;
  
  &::placeholder {
    color: #94a3b8;
  }
  
  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  border: none;
  border-radius: 12px;
  padding: 1rem 1.5rem;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const SuggestionsContainer = styled.div`
  margin-top: 1rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
`;

const SuggestionButton = styled.button`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 20px;
  padding: 0.5rem 1rem;
  color: #ffffff;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }
`;

const LoadingSpinner = styled.div`
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: #ffffff;
  animation: spin 1s ease-in-out infinite;
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`;

const Statistics = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 1rem;
  font-size: 0.8rem;
  color: #94a3b8;
`;

// API Configuration
const API_BASE_URL = 'http://localhost:5001';

interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

function App() {
  const [currentPage, setCurrentPage] = useState('chat');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [stats, setStats] = useState({ messageCount: 0, responseTime: 0 });
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestions = [
    "Hangi algoritma türü hakkında bilgi almak istiyorsun?",
    "Veri analizi için hangi algoritmaları önerirsin?",
    "Machine Learning algoritmaları nelerdir?",
    "Performans optimizasyonu için hangi algoritmalar kullanılır?"
  ];

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Check backend status
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/health`);
        setIsOnline(response.status === 200);
      } catch (error) {
        setIsOnline(false);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Check for saved token on app start
  useEffect(() => {
    const savedToken = localStorage.getItem('accessToken');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      setAccessToken(savedToken);
      setUser(JSON.parse(savedUser));
    }
  }, []);

  const handleLogin = (token: string, userData: any) => {
    setAccessToken(token);
    setUser(userData);
    localStorage.setItem('accessToken', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setCurrentPage('chat');
  };

  const handleLogout = () => {
    setAccessToken(null);
    setUser(null);
    localStorage.removeItem('accessToken');
    localStorage.removeItem('user');
    setCurrentPage('auth');
  };

  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      text: text.trim(),
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    const startTime = Date.now();

    try {
      const endpoint = accessToken ? '/chat/protected' : '/chat';
      const headers = accessToken 
        ? { 'Authorization': `Bearer ${accessToken}` }
        : {};

      const response = await axios.post(
        `${API_BASE_URL}${endpoint}`,
        { message: text.trim() },
        { headers }
      );

      const responseTime = Date.now() - startTime;
      
      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        text: response.data.response || response.data.message || 'Yanıt alınamadı.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
      setStats(prev => ({
        messageCount: prev.messageCount + 1,
        responseTime: responseTime
      }));

    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.',
        isUser: false,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputValue);
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'auth':
        return <Auth onLogin={handleLogin} />;
      
      case 'chat':
        return (
          <>
            <Header>
              <Title>
                <i className="fas fa-brain"></i>
                AI Algoritma Danışmanı
              </Title>
              <Subtitle>
                Yapay zeka destekli algoritma önerileri ve teknik danışmanlık
              </Subtitle>
            </Header>

            <StatusBar>
              <StatusItem>
                <StatusIndicator $isOnline={isOnline} />
                <span>{isOnline ? 'Backend Bağlı' : 'Backend Bağlantısı Yok'}</span>
              </StatusItem>
              <StatusItem>
                <i className="fas fa-message"></i>
                <span>{stats.messageCount} mesaj</span>
              </StatusItem>
              <StatusItem>
                <i className="fas fa-clock"></i>
                <span>{stats.responseTime}ms yanıt süresi</span>
              </StatusItem>
            </StatusBar>

            <ChatContainer>
              <ChatHeader>
                <ChatTitle>
                  <i className="fas fa-comments"></i>
                  Algoritma Danışmanı
                </ChatTitle>
              </ChatHeader>

              <MessagesContainer>
                {messages.length === 0 && (
                  <div style={{ textAlign: 'center', color: '#94a3b8', marginTop: '2rem' }}>
                    <i className="fas fa-robot" style={{ fontSize: '3rem', marginBottom: '1rem' }}></i>
                    <p>Merhaba! Algoritma konusunda size nasıl yardımcı olabilirim?</p>
                  </div>
                )}
                
                {messages.map((message) => (
                  <Message key={message.id} $isUser={message.isUser}>
                    <MessageBubble $isUser={message.isUser}>
                      <MessageText>{message.text}</MessageText>
                    </MessageBubble>
                  </Message>
                ))}
                
                {isLoading && (
                  <Message $isUser={false}>
                    <MessageBubble $isUser={false}>
                      <LoadingSpinner />
                      <span style={{ marginLeft: '0.5rem' }}>Düşünüyor...</span>
                    </MessageBubble>
                  </Message>
                )}
                
                <div ref={messagesEndRef} />
              </MessagesContainer>

              <InputContainer>
                <InputForm onSubmit={handleSubmit}>
                  <InputField
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="Algoritma hakkında soru sorun..."
                    disabled={isLoading}
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage(inputValue);
                      }
                    }}
                  />
                  <SendButton type="submit" disabled={isLoading || !inputValue.trim()}>
                    {isLoading ? (
                      <>
                        <LoadingSpinner />
                        Gönderiliyor...
                      </>
                    ) : (
                      <>
                        <i className="fas fa-paper-plane"></i>
                        Gönder
                      </>
                    )}
                  </SendButton>
                </InputForm>

                <SuggestionsContainer>
                  {suggestions.map((suggestion, index) => (
                    <SuggestionButton
                      key={index}
                      onClick={() => handleSuggestionClick(suggestion)}
                      disabled={isLoading}
                    >
                      {suggestion}
                    </SuggestionButton>
                  ))}
                </SuggestionsContainer>

                <Statistics>
                  <span>Toplam Mesaj: {messages.length}</span>
                  <span>Son Yanıt Süresi: {stats.responseTime}ms</span>
                </Statistics>
              </InputContainer>
            </ChatContainer>
          </>
        );
      
      case 'recommendations':
        return <AlgorithmRecommendation accessToken={accessToken || undefined} />;
      
      case 'profile':
        return accessToken && user ? (
          <UserProfile accessToken={accessToken} user={user} />
        ) : (
          <div style={{ textAlign: 'center', padding: '2rem' }}>
            <p>Profil bilgilerine erişmek için giriş yapın.</p>
          </div>
        );
      
      default:
        return <div>Sayfa bulunamadı.</div>;
    }
  };

  return (
    <AppContainer>
      {currentPage !== 'auth' && (
        <Navigation
          currentPage={currentPage}
          onPageChange={setCurrentPage}
          user={user}
          onLogout={handleLogout}
        />
      )}
      
      <Container>
        {renderPage()}
      </Container>
    </AppContainer>
  );
}

export default App;
