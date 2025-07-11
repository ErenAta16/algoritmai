import React from 'react';
import styled from 'styled-components';

const NavContainer = styled.nav`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 1rem 0;
  margin-bottom: 2rem;
`;

const NavContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Logo = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const NavLinks = styled.div`
  display: flex;
  gap: 1rem;
  align-items: center;
`;

const NavLink = styled.button<{ $active?: boolean }>`
  background: ${props => props.$active ? 'rgba(59, 130, 246, 0.2)' : 'transparent'};
  border: 1px solid ${props => props.$active ? 'rgba(59, 130, 246, 0.3)' : 'rgba(255, 255, 255, 0.1)'};
  border-radius: 8px;
  padding: 0.5rem 1rem;
  color: ${props => props.$active ? '#60a5fa' : '#94a3b8'};
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: rgba(59, 130, 246, 0.1);
    border-color: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
  }
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
`;

const UserName = styled.span`
  color: #94a3b8;
  font-size: 0.9rem;
`;

const LogoutButton = styled.button`
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  padding: 0.5rem 1rem;
  color: #fca5a5;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  &:hover {
    background: rgba(239, 68, 68, 0.2);
    border-color: rgba(239, 68, 68, 0.4);
  }
`;

interface NavigationProps {
  currentPage: string;
  onPageChange: (page: string) => void;
  user?: any;
  onLogout: () => void;
}

const Navigation: React.FC<NavigationProps> = ({ 
  currentPage, 
  onPageChange, 
  user, 
  onLogout 
}) => {
  return (
    <NavContainer>
      <NavContent>
        <Logo>
          <i className="fas fa-brain"></i>
          AI Algoritma Danışmanı
        </Logo>

        <NavLinks>
          <NavLink
            $active={currentPage === 'chat'}
            onClick={() => onPageChange('chat')}
          >
            <i className="fas fa-comments"></i>
            Chat
          </NavLink>

          <NavLink
            $active={currentPage === 'recommendations'}
            onClick={() => onPageChange('recommendations')}
          >
            <i className="fas fa-brain"></i>
            Algoritma Önerileri
          </NavLink>

          {user && (
            <NavLink
              $active={currentPage === 'profile'}
              onClick={() => onPageChange('profile')}
            >
              <i className="fas fa-user"></i>
              Profil
            </NavLink>
          )}
        </NavLinks>

        {user ? (
          <UserInfo>
            <UserName>
              <i className="fas fa-user-circle"></i>
              {' '}{user.username}
            </UserName>
            <LogoutButton onClick={onLogout}>
              <i className="fas fa-sign-out-alt"></i>
              Çıkış
            </LogoutButton>
          </UserInfo>
        ) : (
          <NavLink onClick={() => onPageChange('auth')}>
            <i className="fas fa-sign-in-alt"></i>
            Giriş Yap
          </NavLink>
        )}
      </NavContent>
    </NavContainer>
  );
};

export default Navigation; 