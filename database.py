from datetime import datetime, timezone
import os
from typing import Optional, List
from sqlmodel import Field, SQLModel, create_engine, Session, Relationship


class Company(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_scanned_at: Optional[datetime] = None

    updates: List["CompanyUpdate"] = Relationship(back_populates="company", cascade_delete=True)


class CompanyUpdate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    company_id: int = Field(foreign_key="company.id")
    title: str = Field(index=True)
    url: str
    snippet: Optional[str] = None
    source_type: str = Field(description="official or trusted")
    published_date: Optional[str] = None
    is_read: bool = Field(default=False, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    company: Company = Relationship(back_populates="updates")


# SQLite setup
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
