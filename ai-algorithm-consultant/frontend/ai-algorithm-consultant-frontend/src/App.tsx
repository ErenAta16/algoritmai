import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';
import './App.css';

// Styled components
const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
  font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
`;

const NavBar = styled.nav`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  padding: 16px 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  position: sticky;
  top: 0;
  z-index: 100;
`;

const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  color: white;
  font-size: 24px;
  font-weight: 700;
  cursor: pointer;
`;

const NavButtons = styled.div`
  display: flex;
  gap: 16px;
`;

const NavButton = styled.button<{ active?: boolean }>`
  background: ${props => props.active ? 'rgba(255, 255, 255, 0.2)' : 'transparent'};
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: white;
  padding: 12px 24px;
  border-radius: 12px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.15);
    transform: translateY(-2px);
  }
`;

const HomePage = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
`;

const HeroSection = styled.section`
  padding: 120px 24px 80px;
  text-align: center;
  max-width: 1200px;
  margin: 0 auto;
`;

const HeroTitle = styled.h1`
  color: white;
  font-size: 4rem;
  font-weight: 800;
  margin-bottom: 24px;
  line-height: 1.1;
  letter-spacing: -0.02em;
  
  @media (max-width: 768px) {
    font-size: 2.5rem;
  }
`;

const HeroSubtitle = styled.p`
  color: rgba(255, 255, 255, 0.9);
  font-size: 1.5rem;
  margin-bottom: 32px;
  line-height: 1.6;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
  
  @media (max-width: 768px) {
    font-size: 1.2rem;
  }
`;

const CTAButton = styled.button`
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  color: white;
  padding: 20px 40px;
  border-radius: 16px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 8px 32px rgba(79, 172, 254, 0.4);
  margin: 0 12px;
  
  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(79, 172, 254, 0.6);
  }
`;

const SecondaryButton = styled.button`
  background: transparent;
  border: 2px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 18px 38px;
  border-radius: 16px;
  font-size: 18px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  margin: 0 12px;
  
  &:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-4px);
  }
`;

const FeaturesSection = styled.section`
  padding: 80px 24px;
  max-width: 1200px;
  margin: 0 auto;
`;

const FeaturesGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 32px;
  margin-top: 60px;
`;

const FeatureCard = styled.div`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.12);
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-8px);
    background: rgba(255, 255, 255, 0.12);
  }
`;

const FeatureIcon = styled.div`
  font-size: 48px;
  margin-bottom: 24px;
`;

const FeatureTitle = styled.h3`
  color: white;
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 16px;
`;

const FeatureDescription = styled.p`
  color: rgba(255, 255, 255, 0.8);
  line-height: 1.6;
  font-size: 16px;
`;

const SectionTitle = styled.h2`
  color: white;
  font-size: 3rem;
  font-weight: 700;
  text-align: center;
  margin-bottom: 24px;
`;

const SectionSubtitle = styled.p`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
  text-align: center;
  max-width: 600px;
  margin: 0 auto;
  line-height: 1.6;
`;

const StatsSection = styled.section`
  padding: 80px 24px;
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(20px);
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 40px;
  max-width: 800px;
  margin: 60px auto 0;
`;

const StatCard = styled.div`
  text-align: center;
`;

const StatNumber = styled.div`
  color: #4facfe;
  font-size: 3rem;
  font-weight: 800;
  margin-bottom: 8px;
`;

const StatLabel = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.1rem;
  font-weight: 500;
`;

// Existing chat components (keeping all previous styled components)
const Header = styled.header`
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  padding: 24px;
  text-align: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
`;

const Title = styled.h1`
  color: white;
  margin: 0;
  font-size: 2.2rem;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  letter-spacing: -0.02em;
`;

const Subtitle = styled.p`
  color: rgba(255, 255, 255, 0.8);
  margin: 8px 0 0 0;
  font-size: 1rem;
  font-weight: 400;
`;

const ChatContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
  padding: 24px;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px 0;
  display: flex;
  flex-direction: column;
  gap: 20px;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }
  
  &::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.3);
    border-radius: 3px;
  }
`;

const MessageContainer = styled.div<{ isUser: boolean }>`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  ${props => props.isUser ? 'flex-direction: row-reverse;' : ''}
`;

const MessageBubble = styled.div<{ isUser: boolean }>`
  background: ${props => props.isUser 
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : 'rgba(255, 255, 255, 0.95)'
  };
  color: ${props => props.isUser ? 'white' : '#2d3748'};
  padding: 18px 22px;
  border-radius: 18px;
  max-width: 75%;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
  line-height: 1.6;
  white-space: pre-wrap;
  backdrop-filter: blur(10px);
  border: 1px solid ${props => props.isUser ? 'rgba(255, 255, 255, 0.2)' : 'rgba(255, 255, 255, 0.8)'};
`;

const MessageIcon = styled.div<{ isUser: boolean }>`
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: ${props => props.isUser 
    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
    : 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
  };
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 20px;
  flex-shrink: 0;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
`;

const CodeBlock = styled.div`
  background: #1a202c;
  border: 1px solid #2d3748;
  border-radius: 12px;
  margin: 12px 0;
  overflow: hidden;
  font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
`;

const CodeHeader = styled.div`
  background: #2d3748;
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #4a5568;
`;

const CodeLanguage = styled.span`
  color: #e2e8f0;
  font-size: 14px;
  font-weight: 500;
`;

const CopyButton = styled.button`
  background: #4299e1;
  border: none;
  color: white;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s ease;
  
  &:hover {
    background: #3182ce;
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

const CodeContent = styled.pre`
  color: #e2e8f0;
  padding: 16px;
  margin: 0;
  overflow-x: auto;
  font-size: 14px;
  line-height: 1.5;
  
  /* Custom scrollbar for code */
  &::-webkit-scrollbar {
    height: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: #2d3748;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #4a5568;
    border-radius: 4px;
  }
`;

const SuggestionsContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
`;

const SuggestionButton = styled.button`
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: white;
  padding: 10px 16px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;
  backdrop-filter: blur(10px);
  
  &:hover {
    background: rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
  }
  
  &:active {
    transform: translateY(-1px);
  }
`;

const InputContainer = styled.div`
  display: flex;
  gap: 12px;
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(20px);
  padding: 20px;
  border-radius: 24px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
`;

const Input = styled.input`
  flex: 1;
  border: none;
  background: rgba(255, 255, 255, 0.15);
  color: white;
  padding: 16px 20px;
  border-radius: 20px;
  font-size: 16px;
  outline: none;
  font-family: inherit;
  border: 1px solid rgba(255, 255, 255, 0.1);
  transition: all 0.3s ease;
  
  &::placeholder {
    color: rgba(255, 255, 255, 0.6);
  }
  
  &:focus {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.3);
    box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1);
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
  border: none;
  color: white;
  padding: 16px 24px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 18px;
  font-weight: 600;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(79, 172, 254, 0.4);
  }
  
  &:active {
    transform: translateY(-1px);
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
  }
`;

const LoadingDots = styled.div`
  display: flex;
  gap: 6px;
  padding: 16px 20px;
  
  .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4a5568;
    animation: pulse 1.4s infinite ease-in-out;
  }
  
  .dot:nth-child(1) { animation-delay: -0.32s; }
  .dot:nth-child(2) { animation-delay: -0.16s; }
  .dot:nth-child(3) { animation-delay: 0s; }
  
  @keyframes pulse {
    0%, 80%, 100% { 
      transform: scale(0.8); 
      opacity: 0.5; 
    }
    40% { 
      transform: scale(1.2); 
      opacity: 1; 
    }
  }
`;

const CopyNotification = styled.div<{ show: boolean }>`
  position: fixed;
  top: 20px;
  right: 20px;
  background: #48bb78;
  color: white;
  padding: 12px 20px;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  transform: translateX(${props => props.show ? '0' : '400px'});
  transition: transform 0.3s ease;
  z-index: 1000;
  font-weight: 500;
`;

// Types
interface Message {
  id: string;
  text: string;
  isUser: boolean;
  suggestions?: string[];
  timestamp: Date;
}

interface ChatResponse {
  response: string;
  suggestions?: string[];
}

const API_BASE_URL = 'http://localhost:5000';

// Helper function to render messages with code blocks
const renderMessageContent = (text: string, onCopy: (code: string) => void) => {
  const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(text)) !== null) {
    // Add text before code block
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>
      );
    }

    // Add code block
    const language = match[1] || 'text';
    const code = match[2].trim();
    parts.push(
      <CodeBlock key={`code-${match.index}`}>
        <CodeHeader>
          <CodeLanguage>{language}</CodeLanguage>
          <CopyButton onClick={() => onCopy(code)}>
            📋 Kopyala
          </CopyButton>
        </CodeHeader>
        <CodeContent>{code}</CodeContent>
      </CodeBlock>
    );

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {text.slice(lastIndex)}
      </span>
    );
  }

  return parts.length > 0 ? parts : text;
};

function App() {
  const [currentPage, setCurrentPage] = useState<'home' | 'chat'>('home');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Merhaba! Ben AI Algoritma Danışmanınızım. 🎯\n\nProjeniz için en uygun makine öğrenmesi algoritmalarını bulmanızda yardımcı olacağım. Size özel öneriler geliştirmek için projenizin detaylarını öğrenmek istiyorum.\n\nHangi tür bir makine öğrenmesi projesi üzerinde çalışıyorsunuz?',
      isUser: false,
      suggestions: [
        'Sınıflandırma projesi yapıyorum',
        'Regresyon analizi yapmak istiyorum',
        'Veri kümeleme işlemi yapacağım',
        'Zaman serisi analizi yapıyorum'
      ],
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showCopyNotification, setShowCopyNotification] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleCopyCode = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setShowCopyNotification(true);
      setTimeout(() => setShowCopyNotification(false), 3000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);



  const sendMessage = async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      text: text.trim(),
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      // Prepare conversation history for backend
      const conversationHistory = messages.map(msg => ({
        role: msg.isUser ? 'user' : 'assistant',
        content: msg.text
      }));

      const response = await axios.post<ChatResponse>(`${API_BASE_URL}/chat`, {
        message: text.trim(),
        conversation_history: conversationHistory
      });

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.data.response,
        isUser: false,
        suggestions: response.data.suggestions,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Üzgünüm, şu anda bir bağlantı sorunu yaşıyorum. Lütfen daha sonra tekrar deneyin.',
        isUser: false,
        suggestions: ['Tekrar dene', 'Başka bir soru sor'],
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(inputText);
  };

  const handleSuggestionClick = (suggestion: string) => {
    sendMessage(suggestion);
  };

  return (
    <AppContainer>
      <NavBar>
        <Logo>
          🤖 AI Algoritma Danışmanı
        </Logo>
        <NavButtons>
          <NavButton active={currentPage === 'home'} onClick={() => setCurrentPage('home')}>
            Ana Sayfa
          </NavButton>
          <NavButton active={currentPage === 'chat'} onClick={() => setCurrentPage('chat')}>
            Chat
          </NavButton>
        </NavButtons>
      </NavBar>

      {currentPage === 'home' && (
        <HomePage>
          <HeroSection>
            <HeroTitle>
              AI Algoritma Danışmanı
            </HeroTitle>
            <HeroSubtitle>
              Projeniz için en uygun makine öğrenmesi algoritmalarını bulmanızda yardımcı olacağım.
            </HeroSubtitle>
            <CTAButton onClick={() => setCurrentPage('chat')}>
              Chat'e Başla
            </CTAButton>
            <SecondaryButton onClick={() => setCurrentPage('chat')}>
              Nasıl Çalışır?
            </SecondaryButton>
          </HeroSection>

          <FeaturesSection>
            <SectionTitle>Neler Yapabilirim?</SectionTitle>
            <SectionSubtitle>
              AI Algoritma Danışmanı, makine öğrenmesi projelerinizde size yardımcı olmak için çeşitli özellikler sunar.
            </SectionSubtitle>
            <FeaturesGrid>
              <FeatureCard>
                <FeatureIcon>🔍</FeatureIcon>
                <FeatureTitle>Proje Analizi</FeatureTitle>
                <FeatureDescription>
                  Projenizin detaylarını alıp, en uygun algoritmaları önerebilirim.
                </FeatureDescription>
              </FeatureCard>
              <FeatureCard>
                <FeatureIcon>💡</FeatureIcon>
                <FeatureTitle>Öneri Geliştirme</FeatureTitle>
                <FeatureDescription>
                  Projenizin özelliklerine göre size özel öneriler sunabilirim.
                </FeatureDescription>
              </FeatureCard>
              <FeatureCard>
                <FeatureIcon>🚀</FeatureIcon>
                <FeatureTitle>Hızlı Çözüm</FeatureTitle>
                <FeatureDescription>
                  En hızlı ve etkili çözümü size sunabilirim.
                </FeatureDescription>
              </FeatureCard>
            </FeaturesGrid>
          </FeaturesSection>

          <StatsSection>
            <SectionTitle>İstatistikler</SectionTitle>
            <SectionSubtitle>
              AI Algoritma Danışmanı, kullanıcılarımızın başarılarını gösteriyor.
            </SectionSubtitle>
            <StatsGrid>
              <StatCard>
                <StatNumber>100+</StatNumber>
                <StatLabel>Başarılı Proje</StatLabel>
              </StatCard>
              <StatCard>
                <StatNumber>500+</StatNumber>
                <StatLabel>Önerilen Algoritma</StatLabel>
              </StatCard>
              <StatCard>
                <StatNumber>1000+</StatNumber>
                <StatLabel>Kullanıcı</StatLabel>
              </StatCard>
            </StatsGrid>
          </StatsSection>
        </HomePage>
      )}

      {currentPage === 'chat' && (
        <ChatContainer>
          <Header>
            <Title>
              🤖
              AI Algoritma Danışmanı
            </Title>
            <Subtitle>
              Projeniz için en uygun makine öğrenmesi algoritmalarını bulmanızda yardımcı olacağım.
            </Subtitle>
          </Header>
          
          <MessagesContainer>
            {messages.map((message) => (
              <MessageContainer key={message.id} isUser={message.isUser}>
                <MessageIcon isUser={message.isUser}>
                  {message.isUser ? '👤' : '🤖'}
                </MessageIcon>
                <div>
                  <MessageBubble isUser={message.isUser}>
                    {renderMessageContent(message.text, handleCopyCode)}
                  </MessageBubble>
                  {message.suggestions && message.suggestions.length > 0 && (
                    <SuggestionsContainer>
                      {message.suggestions.map((suggestion, index) => (
                        <SuggestionButton
                          key={index}
                          onClick={() => handleSuggestionClick(suggestion)}
                        >
                          {suggestion}
                        </SuggestionButton>
                      ))}
                    </SuggestionsContainer>
                  )}
                </div>
              </MessageContainer>
            ))}
            
            {isLoading && (
              <MessageContainer isUser={false}>
                <MessageIcon isUser={false}>
                  🤖
                </MessageIcon>
                <MessageBubble isUser={false}>
                  <LoadingDots>
                    <div className="dot"></div>
                    <div className="dot"></div>
                    <div className="dot"></div>
                  </LoadingDots>
                </MessageBubble>
              </MessageContainer>
            )}
            
            <div ref={messagesEndRef} />
          </MessagesContainer>
          
          <form onSubmit={handleSubmit}>
            <InputContainer>
              <Input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Projeniz hakkında bir şeyler yazın..."
                disabled={isLoading}
              />
              <SendButton type="submit" disabled={isLoading || !inputText.trim()}>
                ➤
              </SendButton>
            </InputContainer>
          </form>
        </ChatContainer>
      )}

      {showCopyNotification && (
        <CopyNotification show={showCopyNotification}>
          Kod kopyalandı!
        </CopyNotification>
      )}
    </AppContainer>
  );
}

export default App;
