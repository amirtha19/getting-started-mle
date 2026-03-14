import os
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import HTTPException, FastAPI
from sqlalchemy import text, create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import joblib
import numpy as np
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.logger import get_logger

logger = get_logger(__name__)  
class Settings(BaseSettings):
    database_url: str

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()

engine = create_engine(settings.database_url, connect_args={"sslmode": "require"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

model = None  # global model

class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    prediction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created") 
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris ML API"}

@app.post("/predict")
def predict(data: IrisRequest):
    input_data = np.array([[
        data.sepal_length, data.sepal_width,
        data.petal_length, data.petal_width
    ]])
    prediction = int(model.predict(input_data)[0])
    db = SessionLocal()
    try:
        record = PredictionLog(
            sepal_length=data.sepal_length, sepal_width=data.sepal_width,
            petal_length=data.petal_length, petal_width=data.petal_width,
            prediction=str(prediction)
        )
        db.add(record)
        db.commit()
        logger.info(f"Prediction saved to DB | id={record.id}")
    finally:
        db.close()
    return {"prediction": prediction}

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        logger.error(f"Database unreachable | error={str(e)}")
        raise HTTPException(status_code=500, detail="Database not reachable")
    return {"status": "ok"}