import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ==================================================
# 🔧 CONFIG
# ==================================================

# 데이터 경로
ISSUE_DICT_PATH = "./시사경제용어사전.xlsx"
STAT_DICT_PATH  = "./통계용어사전.xlsx"

# Chroma DB 저장 경로
CHROMA_DB_DIR = "./Chroma_db/Vocab_chroma_db"

# Embedding 모델 이름
EMBEDDING_MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

# ==================================================
# 1. 엑셀 로드
# ==================================================

df_issue = pd.read_excel(ISSUE_DICT_PATH)
df_stat  = pd.read_excel(STAT_DICT_PATH)

documents = []

# ==================================================
# 2. 시사경제용어사전 처리
# ==================================================

for _, row in df_issue.iterrows():
    term = str(row["용어"]).strip()
    desc = str(row["설명"]).strip()
    topic = str(row["주제"]).strip()

    if term:
        documents.append(
            Document(
                page_content=term,   # ✅ 임베딩 대상
                metadata={
                    "description": desc,
                    "source": "시사경제용어사전",
                    "topic": topic
                }
            )
        )

# ==================================================
# 3. 통계용어사전 처리
# ==================================================

for _, row in df_stat.iterrows():
    term = str(row["용어"]).strip()
    desc = str(row["설명"]).strip()

    if term:
        documents.append(
            Document(
                page_content=term,   # ✅ 임베딩 대상
                metadata={
                    "description": desc,
                    "source": "통계용어사전",
                    "topic": "통계"
                }
            )
        )

print(f"총 Document 수: {len(documents)}")

# ==================================================
# 4. Embedding 모델
# ==================================================

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True}
)

# ==================================================
# 5. Chroma DB 저장
# ==================================================

os.makedirs(CHROMA_DB_DIR, exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_DIR,
)

vectorstore.persist()

print("✅ Vocab Chroma DB 저장 완료")
