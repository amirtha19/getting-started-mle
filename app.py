import os
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy import text
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

# -----------------------------
# Database Setup
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float)
    sepal_width = Column(Float)
    petal_length = Column(Float)
    petal_width = Column(Float)
    prediction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


# -----------------------------
# Request Schema
# -----------------------------
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# -----------------------------
# Startup Event
# -----------------------------
@app.on_event("startup")
def startup_event():
    global model
    model = joblib.load("model.pkl")
    print("Model loaded successfully")

    # Create tables if not exists
    Base.metadata.create_all(bind=engine)
    print("Database tables created")


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: IrisRequest):
    input_data = np.array([[ 
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = int(model.predict(input_data)[0])

    # Save to DB
    db = SessionLocal()
    record = PredictionLog(
        sepal_length=data.sepal_length,
        sepal_width=data.sepal_width,
        petal_length=data.petal_length,
        petal_width=data.petal_width,
        prediction=str(prediction)
    )
    db.add(record)
    db.commit()
    db.close()

    return {"prediction": prediction}


@app.get("/health")
def health_check():
    # Check model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Check database connection
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception:
        raise HTTPException(status_code=500, detail="Database not reachable")

    return {"status": "ok"}