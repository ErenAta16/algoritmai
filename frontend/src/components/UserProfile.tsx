import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2rem;
  max-width: 600px;
  margin: 0 auto;
`;

const Title = styled.h2`
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
`;

const ProfileSection = styled.div`
  margin-bottom: 2rem;
`;

const SectionTitle = styled.h3`
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const InfoItem = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1rem;
`;

const InfoLabel = styled.div`
  font-size: 0.8rem;
  color: #64748b;
  font-weight: 500;
  margin-bottom: 0.5rem;
`;

const InfoValue = styled.div`
  font-size: 1rem;
  color: #ffffff;
  font-weight: 500;
`;

const StatusBadge = styled.span<{ $status: string }>`
  background: ${props => {
    switch (props.$status) {
      case 'active': return 'rgba(34, 197, 94, 0.2)';
      case 'inactive': return 'rgba(239, 68, 68, 0.2)';
      case 'pending': return 'rgba(245, 158, 11, 0.2)';
      default: return 'rgba(107, 114, 128, 0.2)';
    }
  }};
  color: ${props => {
    switch (props.$status) {
      case 'active': return '#86efac';
      case 'inactive': return '#fca5a5';
      case 'pending': return '#fcd34d';
      default: return '#9ca3af';
    }
  }};
  border: 1px solid ${props => {
    switch (props.$status) {
      case 'active': return 'rgba(34, 197, 94, 0.3)';
      case 'inactive': return 'rgba(239, 68, 68, 0.3)';
      case 'pending': return 'rgba(245, 158, 11, 0.3)';
      default: return 'rgba(107, 114, 128, 0.3)';
    }
  }};
  border-radius: 20px;
  padding: 0.25rem 0.75rem;
  font-size: 0.8rem;
  font-weight: 500;
`;

const RoleBadge = styled.span<{ $role: string }>`
  background: ${props => {
    switch (props.$role) {
      case 'admin': return 'rgba(168, 85, 247, 0.2)';
      case 'user': return 'rgba(59, 130, 246, 0.2)';
      default: return 'rgba(107, 114, 128, 0.2)';
    }
  }};
  color: ${props => {
    switch (props.$role) {
      case 'admin': return '#c4b5fd';
      case 'user': return '#93c5fd';
      default: return '#9ca3af';
    }
  }};
  border: 1px solid ${props => {
    switch (props.$role) {
      case 'admin': return 'rgba(168, 85, 247, 0.3)';
      case 'user': return 'rgba(59, 130, 246, 0.3)';
      default: return 'rgba(107, 114, 128, 0.3)';
    }
  }};
  border-radius: 20px;
  padding: 0.25rem 0.75rem;
  font-size: 0.8rem;
  font-weight: 500;
`;

const StatsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const StatCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
`;

const StatValue = styled.div`
  font-size: 1.5rem;
  font-weight: 700;
  color: #60a5fa;
  margin-bottom: 0.5rem;
`;

const StatLabel = styled.div`
  font-size: 0.8rem;
  color: #94a3b8;
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

const ErrorMessage = styled.div`
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  padding: 1rem;
  color: #fca5a5;
  font-size: 0.9rem;
  margin-bottom: 1rem;
`;

interface UserProfileProps {
  accessToken: string;
  user: any;
}

const UserProfile: React.FC<UserProfileProps> = ({ accessToken, user }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [userStats, setUserStats] = useState({
    totalMessages: 0,
    totalRecommendations: 0,
    lastLogin: null as string | null
  });

  useEffect(() => {
    if (accessToken && user) {
      fetchUserStats();
    }
  }, [accessToken, user]);

  const fetchUserStats = async () => {
    setLoading(true);
    setError('');

    try {
      // Bu endpoint backend'de mevcut değilse, varsayılan değerler kullanılır
      const response = await axios.get('http://localhost:5001/auth/me', {
        headers: { 'Authorization': `Bearer ${accessToken}` }
      });

      // Gerçek uygulamada bu veriler backend'den gelir
      setUserStats({
        totalMessages: Math.floor(Math.random() * 50) + 10, // Örnek veri
        totalRecommendations: Math.floor(Math.random() * 20) + 5, // Örnek veri
        lastLogin: new Date().toISOString()
      });
    } catch (err: any) {
      setError('Kullanıcı istatistikleri alınamadı.');
      // Hata durumunda varsayılan değerler
      setUserStats({
        totalMessages: 0,
        totalRecommendations: 0,
        lastLogin: null
      });
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Bilinmiyor';
    return new Date(dateString).toLocaleString('tr-TR');
  };

  if (loading) {
    return (
      <Container>
        <Title>
          <i className="fas fa-user"></i>
          Kullanıcı Profili
        </Title>
        <div style={{ textAlign: 'center', padding: '2rem' }}>
          <LoadingSpinner />
          <p style={{ marginTop: '1rem', color: '#94a3b8' }}>Profil yükleniyor...</p>
        </div>
      </Container>
    );
  }

  return (
    <Container>
      <Title>
        <i className="fas fa-user"></i>
        Kullanıcı Profili
      </Title>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      <ProfileSection>
        <SectionTitle>
          <i className="fas fa-info-circle"></i>
          Temel Bilgiler
        </SectionTitle>
        
        <InfoGrid>
          <InfoItem>
            <InfoLabel>Kullanıcı Adı</InfoLabel>
            <InfoValue>{user?.username || 'Bilinmiyor'}</InfoValue>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>E-posta</InfoLabel>
            <InfoValue>{user?.email || 'Bilinmiyor'}</InfoValue>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>Ad Soyad</InfoLabel>
            <InfoValue>{user?.full_name || 'Bilinmiyor'}</InfoValue>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>Kullanıcı ID</InfoLabel>
            <InfoValue style={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>
              {user?.id || 'Bilinmiyor'}
            </InfoValue>
          </InfoItem>
        </InfoGrid>
      </ProfileSection>

      <ProfileSection>
        <SectionTitle>
          <i className="fas fa-shield-alt"></i>
          Hesap Durumu
        </SectionTitle>
        
        <InfoGrid>
          <InfoItem>
            <InfoLabel>Durum</InfoLabel>
            <div style={{ marginTop: '0.5rem' }}>
              <StatusBadge $status={user?.status || 'active'}>
                {user?.status === 'active' ? 'Aktif' : 
                 user?.status === 'inactive' ? 'Pasif' : 
                 user?.status === 'pending' ? 'Beklemede' : 'Bilinmiyor'}
              </StatusBadge>
            </div>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>Rol</InfoLabel>
            <div style={{ marginTop: '0.5rem' }}>
              <RoleBadge $role={user?.role || 'user'}>
                {user?.role === 'admin' ? 'Yönetici' : 
                 user?.role === 'user' ? 'Kullanıcı' : 'Bilinmiyor'}
              </RoleBadge>
            </div>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>Kayıt Tarihi</InfoLabel>
            <InfoValue>{formatDate(user?.created_at)}</InfoValue>
          </InfoItem>
          
          <InfoItem>
            <InfoLabel>Son Güncelleme</InfoLabel>
            <InfoValue>{formatDate(user?.updated_at)}</InfoValue>
          </InfoItem>
        </InfoGrid>
      </ProfileSection>

      <ProfileSection>
        <SectionTitle>
          <i className="fas fa-chart-bar"></i>
          İstatistikler
        </SectionTitle>
        
        <StatsGrid>
          <StatCard>
            <StatValue>{userStats.totalMessages}</StatValue>
            <StatLabel>Toplam Mesaj</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{userStats.totalRecommendations}</StatValue>
            <StatLabel>Algoritma Önerisi</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>{user?.failed_attempts || 0}</StatValue>
            <StatLabel>Başarısız Giriş</StatLabel>
          </StatCard>
          
          <StatCard>
            <StatValue>
              {user?.locked_until ? 'Kilitli' : 'Aktif'}
            </StatValue>
            <StatLabel>Hesap Durumu</StatLabel>
          </StatCard>
        </StatsGrid>
      </ProfileSection>
    </Container>
  );
};

export default UserProfile; 