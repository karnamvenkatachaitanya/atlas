from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from sqlalchemy import event
from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine

DATABASE_URL = "sqlite:///./atlas.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class EpisodeLog(Base):
    __tablename__ = "episode_logs"

    id = Column(Integer, primary_key=True, index=True)
    mode = Column(String, default="startup")
    policy_name = Column(String, default="random")
    created_at = Column(DateTime, default=datetime.utcnow)
    total_reward = Column(Float, default=0.0)
    steps = Column(Integer, default=0)
    final_cash = Column(Float, default=0.0)
    final_revenue = Column(Float, default=0.0)
    summary = Column(JSON, default={})


class StepLog(Base):
    __tablename__ = "step_logs"

    id = Column(Integer, primary_key=True, index=True)
    episode_id = Column(Integer, index=True)
    day = Column(Integer)
    phase = Column(String)
    action = Column(String)
    reward = Column(Float)
    event = Column(JSON, nullable=True)
    state = Column(JSON)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
