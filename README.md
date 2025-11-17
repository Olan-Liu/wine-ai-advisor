# WineAI Advisor

**AI 葡萄酒品質評估與智能推薦系統**  
讓使用者輸入化學數據，即可獲得：
- XGBoost 預測品質分數
- LLM（Llama-3.2）生成品酒師級解釋
- 結果自動儲存 PostgreSQL

---

## 技術棧
| 功能 | 技術 |
|------|------|
| 後端 | FastAPI + SQLAlchemy |
| ML | XGBoost + joblib |
| LLM | Hugging Face Inference API |
| 資料庫 | PostgreSQL |
| 部署 | Docker + docker-compose |

---

## 快速啟動

```bash
# 1. Clone 專案
git clone https://github.com/yourname/wine-ai-advisor.git
cd wine-ai-advisor

# 2. 設定 Hugging Face Token
echo "HF_TOKEN=hf_xxx" > backend/.env

# 3. 訓練模型（第一次）
cd backend
python train_model.py

# 4. 啟動
cd ..
docker-compose up --build
