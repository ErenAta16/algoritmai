import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios';

const Container = styled.div`
  background: rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 2rem;
  margin-bottom: 2rem;
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

const Form = styled.form`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
`;

const InputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const Label = styled.label`
  font-size: 0.9rem;
  font-weight: 500;
  color: #94a3b8;
`;

const Select = styled.select`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 8px;
  padding: 0.75rem;
  color: #ffffff;
  font-size: 1rem;
  cursor: pointer;
  
  &:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
  }
  
  option {
    background: #1e293b;
    color: #ffffff;
  }
`;

const Button = styled.button`
  background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
  border: none;
  border-radius: 8px;
  padding: 0.75rem 1.5rem;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
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

const ResultsContainer = styled.div`
  margin-top: 2rem;
`;

const ResultsTitle = styled.h3`
  font-size: 1.3rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #ffffff;
`;

const AlgorithmCard = styled.div`
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  transition: all 0.2s;
  
  &:hover {
    background: rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
  }
`;

const AlgorithmName = styled.h4`
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #60a5fa;
`;

const AlgorithmDescription = styled.p`
  color: #94a3b8;
  margin-bottom: 1rem;
  line-height: 1.5;
`;

const AlgorithmDetails = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
`;

const DetailItem = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
`;

const DetailLabel = styled.span`
  font-size: 0.8rem;
  color: #64748b;
  font-weight: 500;
`;

const DetailValue = styled.span`
  font-size: 0.9rem;
  color: #ffffff;
  font-weight: 500;
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
  margin-top: 1rem;
`;

const NoResults = styled.div`
  text-align: center;
  color: #94a3b8;
  padding: 2rem;
  font-style: italic;
`;

interface AlgorithmRecommendationProps {
  accessToken?: string;
}

const AlgorithmRecommendation: React.FC<AlgorithmRecommendationProps> = ({ accessToken }) => {
  const [formData, setFormData] = useState({
    problem_type: 'classification',
    data_size: 'medium',
    complexity: 'medium'
  });
  
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState('');

  const handleInputChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResults([]);

    try {
      const endpoint = accessToken ? '/recommend/protected' : '/recommend';
      const headers = accessToken 
        ? { 'Authorization': `Bearer ${accessToken}` }
        : {};

      const response = await axios.post(
        `http://localhost:5001${endpoint}`,
        formData,
        { headers }
      );

      setResults(response.data.recommendations || []);
    } catch (err: any) {
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Algoritma önerisi alınırken bir hata oluştu.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container>
      <Title>
        <i className="fas fa-brain"></i>
        Algoritma Önerisi
      </Title>

      <Form onSubmit={handleSubmit}>
        <InputGroup>
          <Label>Problem Türü</Label>
          <Select
            name="problem_type"
            value={formData.problem_type}
            onChange={handleInputChange}
          >
            <option value="classification">Sınıflandırma</option>
            <option value="regression">Regresyon</option>
            <option value="clustering">Kümeleme</option>
            <option value="dimensionality_reduction">Boyut Azaltma</option>
            <option value="association_rules">İlişki Kuralları</option>
          </Select>
        </InputGroup>

        <InputGroup>
          <Label>Veri Boyutu</Label>
          <Select
            name="data_size"
            value={formData.data_size}
            onChange={handleInputChange}
          >
            <option value="small">Küçük (&lt; 1000)</option>
            <option value="medium">Orta (1000-10000)</option>
            <option value="large">Büyük (&gt; 10000)</option>
          </Select>
        </InputGroup>

        <InputGroup>
          <Label>Karmaşıklık</Label>
          <Select
            name="complexity"
            value={formData.complexity}
            onChange={handleInputChange}
          >
            <option value="simple">Basit</option>
            <option value="medium">Orta</option>
            <option value="complex">Karmaşık</option>
          </Select>
        </InputGroup>

        <Button type="submit" disabled={loading}>
          {loading ? (
            <>
              <LoadingSpinner />
              Öneri Alınıyor...
            </>
          ) : (
            <>
              <i className="fas fa-search"></i>
              Algoritma Önerisi Al
            </>
          )}
        </Button>
      </Form>

      {error && <ErrorMessage>{error}</ErrorMessage>}

      {results.length > 0 && (
        <ResultsContainer>
          <ResultsTitle>
            <i className="fas fa-list"></i>
            Önerilen Algoritmalar ({results.length})
          </ResultsTitle>
          
          {results.map((algorithm, index) => (
            <AlgorithmCard key={index}>
              <AlgorithmName>
                <i className="fas fa-cog"></i>
                {' '}{algorithm.name || algorithm.algorithm_name}
              </AlgorithmName>
              
              <AlgorithmDescription>
                {algorithm.description || algorithm.explanation || 'Açıklama mevcut değil.'}
              </AlgorithmDescription>
              
              <AlgorithmDetails>
                {algorithm.accuracy && (
                  <DetailItem>
                    <DetailLabel>Doğruluk</DetailLabel>
                    <DetailValue>{algorithm.accuracy}%</DetailValue>
                  </DetailItem>
                )}
                
                {algorithm.complexity && (
                  <DetailItem>
                    <DetailLabel>Karmaşıklık</DetailLabel>
                    <DetailValue>{algorithm.complexity}</DetailValue>
                  </DetailItem>
                )}
                
                {algorithm.speed && (
                  <DetailItem>
                    <DetailLabel>Hız</DetailLabel>
                    <DetailValue>{algorithm.speed}</DetailValue>
                  </DetailItem>
                )}
                
                {algorithm.category && (
                  <DetailItem>
                    <DetailLabel>Kategori</DetailLabel>
                    <DetailValue>{algorithm.category}</DetailValue>
                  </DetailItem>
                )}
              </AlgorithmDetails>
            </AlgorithmCard>
          ))}
        </ResultsContainer>
      )}

      {!loading && results.length === 0 && !error && (
        <NoResults>
          <i className="fas fa-info-circle"></i>
          {' '}Algoritma önerisi almak için formu doldurun ve gönderin.
        </NoResults>
      )}
    </Container>
  );
};

export default AlgorithmRecommendation; 