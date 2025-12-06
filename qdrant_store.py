# build_qdrant_store.py

import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

CSV_PATH = "part_01_1.csv"
LOCAL_MODEL_PATH = "E:/Models/frida"

# Qdrant
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "medical_docs_frida"

# Параметры эмбеддера
MAX_LENGTH = 512
POOLING_METHOD = "cls"
BATCH_SIZE = 16
DEVICE = "cuda"


def pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor, method: str = "cls") -> torch.Tensor:
    """Агрегация токенов в один вектор."""
    if method == "mean":
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask
    elif method == "cls":
        return hidden_state[:, 0]
    else:
        raise ValueError(f"Unsupported pooling method: {method}")


def load_csv_chunks(path: str) -> list[Document]:
    """Загружает чанки из CSV и возвращает список Document с метаданными."""
    df = pd.read_csv(path)
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "diagnosis": str(row.get("diagnosis", ""))[:500],
            "category": str(row.get("parent_section_title", ""))[:256],
            "url": str(row.get("url", ""))[:512],
            "path": str(row.get("path", ""))[:512],
            "chunk_index": str(row.get("refined_chunk_index", ""))[:100],
        }
        content = str(row["chunk_text"]) if pd.notna(row["chunk_text"]) else ""
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


class FridaEmbedder:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 512, pooling: str = "cls"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.model = T5EncoderModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.pooling = pooling

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = pool(outputs.last_hidden_state, inputs["attention_mask"], self.pooling)
                emb = F.normalize(emb, p=2, dim=1)
                all_embeddings.extend(emb.cpu().float().numpy().tolist())
        return all_embeddings


if __name__ == "__main__":
    print("-- Загрузка документов...")
    documents = load_csv_chunks(CSV_PATH)
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    print(f"-- Загружено {len(documents)} документов")

    print("-- Инициализация Frida-эмбеддера...")
    embedder = FridaEmbedder(
        model_path=LOCAL_MODEL_PATH,
        device=DEVICE,
        max_length=MAX_LENGTH,
        pooling=POOLING_METHOD
    )

    print("-- Генерация эмбеддингов...")
    embeddings = embedder.embed(texts)
    dim = len(embeddings[0])
    print(f"-- Эмбеддинги готовы. Размерность: {dim}")

    print("-- Подключение к Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Удалим коллекцию, если она существует (для перезапуска)
    if client.collection_exists(COLLECTION_NAME):
        print(f"+ Коллекция '{COLLECTION_NAME}' уже существует — пересоздаём.")
        client.delete_collection(COLLECTION_NAME)

    print("-- Создание коллекции в Qdrant...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    print("-- Загрузка данных в Qdrant...")
    points = []
    for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
        # Обрезаем текст, чтобы избежать ошибок при очень длинных чанках
        safe_text = text[:2000] if len(text) > 2000 else text
        points.append(
            PointStruct(
                id=i + 1,
                vector=embedding,
                payload={
                    "text": safe_text,
                    **metadata
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("🎉 Данные успешно загружены в Qdrant!")
    print(f"🔗 Откройте http://localhost:6333/dashboard для просмотра (если Qdrant запущен локально).")