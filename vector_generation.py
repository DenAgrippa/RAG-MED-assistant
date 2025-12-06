# build_vectorstore.py

import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, T5EncoderModel
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

CSV_PATH = "part_01_1.csv"
LOCAL_MODEL_PATH = "E:/Models/frida"
INDEX_DIR = "F:/FAISS_index/vectorstores/faiss_store"

os.makedirs(INDEX_DIR, exist_ok=True)

# Вспомогательные функции

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
            "diagnosis": row.get("diagnosis", ""),
            "category": row.get("parent_section_title", ""),
            "url": row.get("url", ""),
            "path": row.get("path", ""),
            "chunk_index": row.get("refined_chunk_index", ""),
        }
        content = str(row["chunk_text"]) if pd.notna(row["chunk_text"]) else ""
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# FridaEmbeddings — кастомный эмбеддер
class FridaEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        pooling_method: str = "cls",
        normalize_embeddings: bool = True,
        max_length: int = 512,
        batch_size: int = 16,
        device: str = "cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True
        )
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device).eval()
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Обработка батчами
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                    method=self.pooling_method
                )
                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().float().numpy().tolist())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# Основной рабочий процесс
if __name__ == "__main__":
    print("-- Загрузка документов...")
    documents = load_csv_chunks(CSV_PATH)
    print(f"-- Загружено {len(documents)} документов")

    index_file = os.path.join(INDEX_DIR, "index.faiss")
    store_file = os.path.join(INDEX_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(store_file):
        print("-- Индекс уже существует — пропускаем создание")
    else:
        print("-- Инициализация Frida-эмбеддера...")
        embedding_model = FridaEmbeddings(
            model_name=LOCAL_MODEL_PATH,
            pooling_method="cls",
            normalize_embeddings=True,
            max_length=512,
            batch_size=16,
            device="cuda"
        )

        print("-- Проверка эмбеддера...")
        test_vec = embedding_model.embed_query("Тестовый запрос")
        print(f"-- Размерность эмбеддинга: {len(test_vec)}")

        print("-- Построение FAISS-индекса...")
        index = faiss.IndexFlatIP(len(test_vec))
        vectorstore = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={}
        )

        print("-- Добавление документов в индекс (батчами)...")
        vectorstore.add_documents(documents)
        print("-- Сохранение индекса...")
        vectorstore.save_local(INDEX_DIR)

    print("-- Векторная база данных успешно создана и сохранена!")