from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib
from pathlib import Path
import traceback
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# ================== 載入環境變數 ==================
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HF_TOKEN in .env file")

# ================== 路徑 ==================
BASE_DIR = Path(__file__).parent.resolve()
MODEL_PATH = BASE_DIR / "model.pkl"

# ================== FastAPI ==================
app = FastAPI()


class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# ================== 資料庫 ==================
DATABASE_URL = "postgresql://postgres:password@localhost:5432/wine_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    fixed_acidity = Column(Float)
    volatile_acidity = Column(Float)
    citric_acid = Column(Float)
    residual_sugar = Column(Float)
    chlorides = Column(Float)
    free_sulfur_dioxide = Column(Float)
    total_sulfur_dioxide = Column(Float)
    density = Column(Float)
    pH = Column(Float)
    sulphates = Column(Float)
    alcohol = Column(Float)
    quality = Column(Integer)
    explanation = Column(String)


Base.metadata.create_all(bind=engine)

# ================== 載入 ML 模型 ==================
model = None
try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print(f"ML Model loaded: {MODEL_PATH}")
    else:
        print("model.pkl not found! Using dummy quality=6.")
except Exception as e:
    print(f"ML Model load failed: {e}")
    traceback.print_exc()

# ================== 初始化 Hugging Face LLM ==================
client = InferenceClient(token=HF_TOKEN)

# ================== API ==================


@app.post("/predict")
async def predict_quality(data: WineData):
    try:
        # 1. 特徵準備（修正：用 numpy array 確保格式）
        import numpy as np
        features = np.array([[
            data.fixed_acidity, data.volatile_acidity, data.citric_acid,
            data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
            data.total_sulfur_dioxide, data.density, data.pH,
            data.sulphates, data.alcohol
        ]])

        # 2. ML 預測品質（修正：加錯誤處理，避免 -1）
        if model is not None:
            pred = model.predict(features)[0]
            quality = max(3, min(10, int(round(pred))))  # 限制在 3-10，避免異常
            print(f"ML Prediction: {pred} → {quality}")
        else:
            quality = 6

        # 3. LLM 生成解釋（修正：用 conversational task + 新模型）
        prompt = f"You are a professional sommelier. Explain this wine in 1-2 elegant sentences: Quality {quality}/10, Alcohol {data.alcohol}%, Acidity {data.fixed_acidity}, Sugar {data.residual_sugar}g/L, pH {data.pH}."

        try:
            # 換模型：Llama-3.2-1B-Instruct (支援 text-gen + conversational)
            llm_response = client.conversational(
                model="meta-llama/Llama-3.2-1B-Instruct",  # 免費、快速、支援 instruct
                messages=[{"role": "user", "content": prompt}],
                max_new_tokens=80,
                temperature=0.7
            )
            explanation = llm_response.generated_text  # 取生成文字
            print(f"LLM Response: {explanation[:100]}...")  # 印前 100 字除錯
        except Exception as e:
            explanation = f"LLM fallback: This wine scores {quality}/10 with balanced {data.alcohol}% alcohol and {data.fixed_acidity} acidity."
            print(f"LLM Error fallback: {e}")

        # 4. 存入資料庫
        db = SessionLocal()
        pred_record = Prediction(
            **data.dict(),
            quality=quality,
            explanation=explanation
        )
        db.add(pred_record)
        db.commit()
        db.refresh(pred_record)
        db.close()

        return {
            "quality": quality,
            "explanation": explanation,
            "db_id": pred_record.id
        }

    except Exception as e:
        print("API Error:")
        traceback.print_exc()
        return {"error": str(e)}, 500
