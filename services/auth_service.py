"""
Comprehensive Authentication Service
JWT Token Management, Password Hashing, User Management, Role-Based Access Control
"""

import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from enum import Enum
import json
import hashlib
import secrets
from uuid import uuid4

from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, Field, validator, EmailStr
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from db.models import User, UserRoleEnum, UserStatusEnum
from db.session import SessionLocal

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()

class UserRole(str, Enum):
    """User roles for role-based access control"""
    ANONYMOUS = "anonymous"
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    MODERATOR = "moderator"

class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class TokenType(str, Enum):
    """Token types"""
    ACCESS = "access"
    REFRESH = "refresh"

class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    
    @validator('username')
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserLogin(BaseModel):
    """User login model"""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")

class UserUpdate(BaseModel):
    """User update model"""
    full_name: Optional[str] = Field(None, max_length=100)
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None

class UserResponse(BaseModel):
    """User response model"""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

class TokenData(BaseModel):
    """Token data model"""
    user_id: str
    username: str
    role: UserRole
    token_type: TokenType
    exp: datetime

class AuthService:
    """
    SQLAlchemy tabanlı authentication service
    """
    def __init__(self):
        self.failed_login_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300
        self.refresh_tokens_db = {}  # In-memory for now, can be moved to DB later
        self._create_default_admin()
        logger.info("✅ Authentication service (DB) initialized")

    def get_db(self):
        """Database session dependency"""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()



    def get_user_by_username(self, db: Session, username: str) -> Optional[User]:
        """Get user by username from database"""
        return db.query(User).filter(User.username == username).first()

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Get user by email from database"""
        return db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, db: Session, user_id: str) -> Optional[User]:
        """Get user by ID from database"""
        return db.query(User).filter(User.id == user_id).first()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash password"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire, 
            "type": TokenType.ACCESS,
            "user_id": data.get("sub"),
            "token_type": TokenType.ACCESS
        })
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({
            "exp": expire, 
            "type": TokenType.REFRESH,
            "user_id": data.get("sub"),
            "token_type": TokenType.REFRESH
        })
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            token_data = TokenData(**payload)
            # Check if token is expired
            now_ts = datetime.utcnow().timestamp()
            exp_ts = token_data.exp.timestamp() if hasattr(token_data.exp, "timestamp") else float(token_data.exp)
            if now_ts > exp_ts:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            return token_data
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
    
    def create_user(self, db: Session, username: str, email: str, password: str, 
                   full_name: Optional[str] = None, role: UserRole = UserRole.USER) -> UserResponse:
        """Create new user in database"""
        # Check if user already exists
        if self.get_user_by_username(db, username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        if self.get_user_by_email(db, email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user
        hashed_password = self.get_password_hash(password)
        
        db_user = User(
            id=uuid4().hex,
            username=username,
            email=email,
            full_name=full_name,
            hashed_password=hashed_password,
            role=UserRoleEnum.user,
            status=UserStatusEnum.active,
            created_at=datetime.utcnow(),
            last_login=None,
            failed_attempts=0,
            locked_until=None
        )
        
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        logger.info(f"✅ User created: {username} (ID: {db_user.id})")
        
        return UserResponse.from_orm(db_user)
    
    def authenticate_user(self, db: Session, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password"""
        # Check if user is locked
        if self._is_user_locked(username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to too many failed attempts"
            )
        
        # Find user by username or email
        user = self.get_user_by_username(db, username)
        if not user:
            user = self.get_user_by_email(db, username)
        
        if not user:
            self._record_failed_attempt(username)
            return None
        
        # Verify password
        if not self.verify_password(password, user.hashed_password):
            self._record_failed_attempt(username)
            return None
        
        # Reset failed attempts on successful login
        user.failed_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        db.commit()
        
        logger.info(f"✅ User authenticated: {username}")
        
        return user
    
    def login(self, db: Session, username: str, password: str) -> TokenResponse:
        """User login and token generation"""
        user = self.authenticate_user(db, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        # Check if user is active
        if user.status != UserStatusEnum.active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not active"
            )
        
        # Create tokens
        access_token = self.create_access_token(
            data={"sub": user.id, "username": user.username, "role": user.role.value}
        )
        refresh_token = self.create_refresh_token(
            data={"sub": user.id, "username": user.username, "role": user.role.value}
        )
        
        # Store refresh token
        self.refresh_tokens_db[user.id] = refresh_token
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.from_orm(user)
        )
    
    def refresh_token(self, db: Session, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        try:
            token_data = self.verify_token(refresh_token)
            
            if token_data.token_type != TokenType.REFRESH:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Check if refresh token is stored
            stored_token = self.refresh_tokens_db.get(token_data.user_id)
            if not stored_token or stored_token != refresh_token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            # Get user
            user = self.get_user_by_id(db, token_data.user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            # Create new tokens
            new_access_token = self.create_access_token(
                data={"sub": user.id, "username": user.username, "role": user.role.value}
            )
            new_refresh_token = self.create_refresh_token(
                data={"sub": user.id, "username": user.username, "role": user.role.value}
            )
            
            # Update stored refresh token
            self.refresh_tokens_db[user.id] = new_refresh_token
            
            return TokenResponse(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                user=UserResponse.from_orm(user)
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not refresh token"
            )
    
    def logout(self, user_id: str) -> bool:
        """User logout - invalidate refresh token"""
        if user_id in self.refresh_tokens_db:
            del self.refresh_tokens_db[user_id]
            logger.info(f"✅ User logged out: {user_id}")
            return True
        return False
    
    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(lambda: next(self.get_db())) ) -> UserResponse:
        """Get current authenticated user"""
        token = credentials.credentials
        token_data = self.verify_token(token)
        if token_data.token_type != TokenType.ACCESS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        user = self.get_user_by_id(db, token_data.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        if user.status != UserStatusEnum.active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not active"
            )
        return UserResponse.from_orm(user)

    def get_current_user_dependency(self):
        """Dependency function for getting current user"""
        def _get_current_user(
            credentials: HTTPAuthorizationCredentials = Depends(security),
            db: Session = Depends(self.get_db)
        ) -> UserResponse:
            token = credentials.credentials
            token_data = self.verify_token(token)
            if token_data.token_type != TokenType.ACCESS:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            user = self.get_user_by_id(db, token_data.user_id)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            if user.status != UserStatusEnum.active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is not active"
                )
            return UserResponse.from_orm(user)
        return _get_current_user

    def require_role(self, required_roles: List[UserRole]):
        """Dependency for role-based access control"""
        def role_checker(
            current_user: UserResponse = Depends(self.get_current_user_dependency()),
            db: Session = Depends(self.get_db)
        ):
            if current_user.role not in required_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return current_user
        return role_checker
    
    def update_user(self, db: Session, user_id: str, update_data: UserUpdate) -> UserResponse:
        """Update user information"""
        user = self.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update fields
        if update_data.full_name is not None:
            user.full_name = update_data.full_name
        if update_data.email is not None:
            # Check if email is already taken
            existing_user = self.get_user_by_email(db, update_data.email)
            if existing_user and existing_user.id != user_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken"
                )
            user.email = update_data.email
        if update_data.role is not None:
            user.role = UserRoleEnum(update_data.role.value)
        if update_data.status is not None:
            user.status = UserStatusEnum(update_data.status.value)
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ User updated: {user.username}")
        
        return UserResponse.from_orm(user)
    
    def change_password(self, db: Session, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password"""
        user = self.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        user.hashed_password = self.get_password_hash(new_password)
        db.commit()
        
        logger.info(f"✅ Password changed for user: {user.username}")
        
        return True
    
    def delete_user(self, db: Session, user_id: str) -> bool:
        """Delete user account"""
        user = self.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        username = user.username
        
        # Remove user
        db.delete(user)
        db.commit()
        
        # Remove refresh token
        if user_id in self.refresh_tokens_db:
            del self.refresh_tokens_db[user_id]
        
        logger.info(f"✅ User deleted: {username}")
        
        return True
    
    def get_all_users(self, db: Session, current_user: UserResponse) -> List[UserResponse]:
        """Get all users (admin only)"""
        if current_user.role not in [UserRole.ADMIN, UserRole.MODERATOR]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        users = db.query(User).all()
        return [UserResponse.from_orm(user) for user in users]
    
    def _create_default_admin(self):
        """Create default admin user if not exists"""
        db = SessionLocal()
        try:
            # Check if admin user exists
            admin_user = self.get_user_by_username(db, "admin")
            if not admin_user:
                # Create default admin user
                admin_password = os.getenv("ADMIN_PASSWORD", "Admin123!")
                self.create_user(
                    db=db,
                    username="admin",
                    email="admin@algoritma.com",
                    password=admin_password,
                    full_name="System Administrator",
                    role=UserRoleEnum.admin
                )
                logger.info("✅ Default admin user created: admin")
        except Exception as e:
            logger.error(f"❌ Error creating default admin: {e}")
        finally:
            db.close()
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username in self.failed_login_attempts:
            self.failed_login_attempts[username]['count'] += 1
            self.failed_login_attempts[username]['last_attempt'] = datetime.utcnow()
        else:
            self.failed_login_attempts[username] = {
                'count': 1,
                'last_attempt': datetime.utcnow()
            }
        
        # Update user data in database
        db = SessionLocal()
        try:
            user = self.get_user_by_username(db, username)
            if user:
                user.failed_attempts = self.failed_login_attempts[username]['count']
                if self.failed_login_attempts[username]['count'] >= self.max_failed_attempts:
                    lockout_until = datetime.utcnow() + timedelta(seconds=self.lockout_duration)
                    user.locked_until = lockout_until
                    self.failed_login_attempts[username]['locked_until'] = lockout_until
                db.commit()
        except Exception as e:
            logger.error(f"❌ Error updating failed attempts: {e}")
        finally:
            db.close()
        
        logger.warning(f"⚠️ Failed login attempt for user: {username}")
    
    def _is_user_locked(self, username: str) -> bool:
        """Check if user is locked due to failed attempts"""
        if username not in self.failed_login_attempts:
            return False
        
        attempt_data = self.failed_login_attempts[username]
        
        # Check if user is locked
        if 'locked_until' in attempt_data and attempt_data['locked_until']:
            if datetime.utcnow() < attempt_data['locked_until']:
                return True
            else:
                # Unlock user
                attempt_data['count'] = 0
                attempt_data['locked_until'] = None
                
                # Update database
                db = SessionLocal()
                try:
                    user = self.get_user_by_username(db, username)
                    if user:
                        user.failed_attempts = 0
                        user.locked_until = None
                        db.commit()
                except Exception as e:
                    logger.error(f"❌ Error unlocking user: {e}")
                finally:
                    db.close()
        
        return False
    
    def get_auth_stats(self, db: Session) -> Dict:
        """Get authentication statistics"""
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.status == UserStatusEnum.ACTIVE).count()
        locked_users = db.query(User).filter(
            User.locked_until.isnot(None),
            User.locked_until > datetime.utcnow()
        ).count()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'locked_users': locked_users,
            'failed_attempts': len(self.failed_login_attempts),
            'active_sessions': len(self.refresh_tokens_db)
        }

# Global authentication service instance
auth_service = AuthService() 