from sqlalchemy import Column, String, DateTime, Enum, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()

class UserRoleEnum(str, enum.Enum):
    anonymous = "anonymous"
    user = "user"
    premium = "premium"
    admin = "admin"
    moderator = "moderator"

class UserStatusEnum(str, enum.Enum):
    active = "active"
    inactive = "inactive"
    suspended = "suspended"
    pending = "pending"

class User(Base):
    __tablename__ = "users"
    id = Column(String(32), primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(128), nullable=False)
    role = Column(Enum(UserRoleEnum), default=UserRoleEnum.user, nullable=False)
    status = Column(Enum(UserStatusEnum), default=UserStatusEnum.active, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True) 