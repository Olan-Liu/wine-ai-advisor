# ... (原有 import)
import xgboost as xgb
from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 載入你原有 XGBoost 模型 (假設存為 model.xgb)
# 用 joblib.load 如果是 scikit-learn wrapper
model = xgb.Booster(model_file='path/to/your/model.xgb')

# DB 連線
engine = create_engine('postgresql://postgres:password@localhost:5432/wine_db')
Base = declarative_base()
Session = sessionmaker(bind=engine)


class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    fixed_acidity = Column(Float)
    # ... 其他欄位
    quality = Column(Integer)
    explanation = Column(String)


Base.metadata.create_all(engine)


@app.post("/predict")
async def predict_quality(data: WineData):
    # 轉成 XGBoost 輸入
    features = [[data.fixed_acidity, data.volatile_acidity, ...]]  # 完整 11 特徵
    dmatrix = xgb.DMatrix(features)
    quality = int(model.predict(dmatrix)[0])  # 預測

    # 暫存 explanation
    explanation = "Placeholder"

    # 存 DB
    session = Session()
    pred = Prediction(fixed_acidity=data.fixed_acidity, ..., quality=quality, explanation=explanation)
    session.add(pred)
    session.commit()
    session.close()

    return {"quality": quality, "explanation": explanation}
