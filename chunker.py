import json
import sqlite3
import tiktoken
from chonkie import SentenceChunker, OverlapRefinery
from typing import List, Dict, Any, Union
import os
import pandas as pd
# Добавим nltk для более точного разбиения по предложениям, если нужно.
# pip install nltk
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading punkt tokenizer for sentence splitting...")
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# --- 1. Функция для подсчета токенов с помощью tiktoken ---
def count_tokens_tiktoken(text: str, encoding) -> int:
    """Функция, которая возвращает количество токенов в тексте."""
    if not text:
        return 0
    return len(encoding.encode(text))

# --- 2. Улучшенная функция для разбиения content с учетом предложений и токенов ---
def chunk_content_if_needed(content_text: str, max_tokens: int, enc) -> List[Dict[str, Union[str, int]]]:
    """
    Проверяет размер content_text. Если он превышает max_tokens,
    разбивает его на предложения, аккумулируя до max_tokens.
    Если одно предложение превышает max_tokens, применяет smart_chunk_by_tokens_with_sentence_boundary.
    Возвращает список словарей {'text': str, 'token_count': int}.
    """
    if not content_text or len(content_text.strip()) == 0:
        return [{"text": content_text, "token_count": 0}]

    initial_token_count = count_tokens_tiktoken(content_text, enc)

    # Если content короче лимита, возвращаем его как один чанк
    if initial_token_count <= max_tokens:
        return [{"text": content_text, "token_count": initial_token_count}]

    # Если content длиннее лимита, разбиваем на предложения вручную или с помощью nltk
    # Chonkie SentenceChunker здесь используется для получения Chunk-объектов
    sentences = sent_tokenize(content_text, language='russian')

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_token_count = count_tokens_tiktoken(sentence, enc)

        # Проверяем, уложится ли предложение в лимит, если добавить его к текущему чанку
        potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
        potential_chunk_tokens = count_tokens_tiktoken(potential_chunk, enc)

        if potential_chunk_tokens <= max_tokens:
            # Добавляем предложение к текущему чанку
            current_chunk = potential_chunk
            current_chunk_tokens = potential_chunk_tokens
        else:
            # Текущий чанк готов (если не пустой)
            if current_chunk.strip():
                chunks.append({"text": current_chunk, "token_count": current_chunk_tokens})

            # Проверяем, не превышает ли само предложение лимит
            if sentence_token_count > max_tokens:
                # Разбиваем длинное предложение с учетом границ
                sub_chunks = smart_chunk_by_tokens_with_sentence_boundary(sentence, max_tokens, enc)
                chunks.extend(sub_chunks)
            else:
                # Начинаем новый чанк с этого предложения
                current_chunk = sentence
                current_chunk_tokens = sentence_token_count

    # Добавляем последний чанк, если он не пуст
    if current_chunk.strip():
        chunks.append({"text": current_chunk, "token_count": current_chunk_tokens})

    return chunks


# --- 3. Улучшенная функция разбиения по токенам с учетом границ предложений ---
def smart_chunk_by_tokens_with_sentence_boundary(text: str, max_tokens: int, encoding) -> List[Dict[str, Union[str, int]]]:
    """
    Разбивает текст на фрагменты, каждый из которых не превышает max_tokens.
    Старается разорвать фрагмент на границе предложения (. ! ?) до max_tokens, если возможно.
    Использует tiktoken для подсчета и разбиения.
    Возвращает список словарей {'text': str, 'token_count': int}.
    """
    if not text or max_tokens <= 0:
        return [{"text": text, "token_count": count_tokens_tiktoken(text, encoding)}] if text else []

    tokens = encoding.encode(text)
    chunks = []
    start_idx = 0

    while start_idx < len(tokens):
        end_idx = start_idx + max_tokens

        # Если это конец текста, просто добавляем остаток (даже если превышает лимит)
        if end_idx >= len(tokens):
            chunk_tokens = tokens[start_idx:]
            chunk_text = encoding.decode(chunk_tokens)
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "token_count": len(chunk_tokens)})
            break

        # Пытаемся разорвать на границе предложения внутри среза [start_idx : end_idx]
        chunk_tokens = tokens[start_idx:end_idx]
        potential_chunk_text = encoding.decode(chunk_tokens)

        # Ищем последний подходящий знак препинания в potential_chunk_text
        last_punct_index = -1
        for punct in ['.', '!', '?']:
            idx = potential_chunk_text.rfind(punct)
            if idx > last_punct_index:
                last_punct_index = idx

        final_chunk_text = potential_chunk_text
        if last_punct_index != -1:
            # Нашли потенциальную границу
            candidate_text = potential_chunk_text[:last_punct_index+1] # Включаем знак препинания
            # Проверяем, укладывается ли кандидат в токены
            candidate_tokens = encoding.encode(candidate_text)
            if len(candidate_tokens) <= max_tokens:
                # Кандидат подходит, используем его
                final_chunk_text = candidate_text
                actual_tokens_used = len(candidate_tokens)
            else:
                # Кандидат не укладывается, используем грубое разбиение
                actual_tokens_used = max_tokens
        else:
            # Не найдено знаков препинания до max_tokens, используем грубое разбиение
            actual_tokens_used = max_tokens

        if actual_tokens_used <= 0:
            # Защита от зацикливания, если нечего добавить
            start_idx += 1
            continue

        if final_chunk_text.strip():
            chunks.append({"text": final_chunk_text, "token_count": actual_tokens_used})

        # Сдвигаем start_idx на основе *фактически использованных токенов*
        # Если использовали грубое разбиение (actual_tokens_used == max_tokens)
        # start_idx += actual_tokens_used
        # Если использовали разбиение по предложению (actual_tokens_used < max_tokens)
        # start_idx += actual_tokens_used
        # В обоих случаях сдвигаем на actual_tokens_used
        start_idx += actual_tokens_used

    return chunks


# --- 4. Рекурсивная функция для обхода JSON и сбора "глубоких" content ---
def collect_leaf_content_recursive(sections: List[Dict], diagnosis: str, url: str, all_chunks: List[Dict], max_tokens: int, enc, parent_meta: Dict[str, Any] = None, stats: Dict[str, Any] = None):
    """
    Рекурсивно обходит список разделов и их подразделов,
    находит "листья" (разделы без subsections) и разбивает их content при необходимости.
    Обновляет словарь stats с информацией о процессе, используя токены.
    """
    if parent_meta is None:
        parent_metadata = {}
    else:
        parent_metadata = parent_meta

    if stats is None:
        stats = {
            "total_leaf_sections_processed": 0,
            "total_chunks_created_before_refinery": 0,
            "total_content_tokens_processed": 0,
            "chunk_lengths_tokens_before_refinery": [],
            "max_tokens_per_chunk_setting": max_tokens,
        }

    for section in sections:
        title = section.get("title", "")
        section_id = section.get("id", "")
        level = section.get("level", 0)
        content = section.get("content", "")
        subsections = section.get("subsections", [])

        current_metadata = {
            "diagnosis": diagnosis,
            "url": url,
            "section_title": title,
            "section_id": section_id,
            "section_level": level,
            **{k: v for k, v in parent_metadata.items() if not k.startswith('current_')}
        }

        if subsections:
            collect_leaf_content_recursive(subsections, diagnosis, url, all_chunks, max_tokens, enc, parent_meta=current_metadata, stats=stats)
        else:
            stats["total_leaf_sections_processed"] = stats.get("total_leaf_sections_processed", 0) + 1
            content_tokens_count = count_tokens_tiktoken(content, enc)
            stats["total_content_tokens_processed"] = stats.get("total_content_tokens_processed", 0) + content_tokens_count

            # Разбиваем content при необходимости, используя улучшенную логику
            processed_chunks = chunk_content_if_needed(content, max_tokens, enc)

            for i, chunk_info in enumerate(processed_chunks):
                 chunk_text = chunk_info["text"]
                 chunk_tokens_count = chunk_info["token_count"]

                 stats["total_chunks_created_before_refinery"] = stats.get("total_chunks_created_before_refinery", 0) + 1
                 stats["chunk_lengths_tokens_before_refinery"].append(chunk_tokens_count)

                 chunk_entry = {
                    "diagnosis": diagnosis,
                    "url": url,
                    "parent_section_title": parent_metadata.get("section_title", ""),
                    "parent_section_id": parent_metadata.get("section_id", ""),
                    "parent_section_level": parent_metadata.get("section_level", 0),
                    "leaf_title": title,
                    "leaf_id": section_id,
                    "leaf_level": level,
                    "chunk_index": i,
                    "content": chunk_text,
                    "token_count": chunk_tokens_count
                }
                 all_chunks.append(chunk_entry)

# --- 5. Функция для вычисления статистики ---
def calculate_chunk_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вычисляет основные статистики на основе собранных данных (до или после применения Refinery).
    """
    # Выбираем ключ для длин чанков в зависимости от того, какая статистика передана
    # Предполагаем, что ключи могут быть как "before", так и "after"
    lengths_key = "chunk_lengths_tokens_before_refinery"
    if "chunk_lengths_tokens_after_refinery" in stats:
        lengths_key = "chunk_lengths_tokens_after_refinery"

    lengths = stats.get(lengths_key, [])
    if not lengths:
        return {
            "avg_chunk_length_tokens": 0,
            "min_chunk_length_tokens": 0,
            "max_chunk_length_tokens": 0,
            "total_chunks": stats.get("total_chunks_created_before_refinery", 0) + stats.get("total_chunks_after_refinery", 0),
            "total_leaf_sections_processed": stats.get("total_leaf_sections_processed", 0),
            "total_content_tokens_processed": stats.get("total_content_tokens_processed", 0),
            "max_tokens_per_chunk_setting": stats.get("max_tokens_per_chunk_setting", "unknown"),
            "overlap_context_size_fraction_used": stats.get("overlap_context_size_fraction_used", "N/A"),
        }

    avg_len = sum(lengths) / len(lengths) if len(lengths) > 0 else 0
    min_len = min(lengths)
    max_len = max(lengths)

    # Возвращаем статистику с ключами, указывающими, к какому этапу она относится
    # Используем общий ключ для количества чанков, выбирая из доступных
    total_chunks_key = "total_chunks_created_before_refinery"
    if "total_chunks_after_refinery" in stats:
        total_chunks_key = "total_chunks_after_refinery"

    return {
        "avg_chunk_length_tokens": round(avg_len, 2),
        "min_chunk_length_tokens": min_len,
        "max_chunk_length_tokens": max_len,
        "total_chunks": stats.get(total_chunks_key, 0),
        "total_leaf_sections_processed": stats.get("total_leaf_sections_processed", 0),
        "total_content_tokens_processed": stats.get("total_content_tokens_processed", 0),
        "max_tokens_per_chunk_setting": stats.get("max_tokens_per_chunk_setting", "unknown"),
        "overlap_context_size_fraction_used": stats.get("overlap_context_size_fraction_used", "N/A"),
    }


# --- 6. Основная функция для работы с БД и сохранения в CSV ---
def process_kr_from_db(db_path: str, output_csv_name: str, output_json_name: str, max_tokens_per_content_chunk: int = 384, overlap_tokens: int = 30):
    """
    Подключается к базе данных SQLite, извлекает данные из таблицы `cr`, чанкует их и сохраняет результат в CSV и JSON.
    """
    # Подключение к базе данных
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Запрос для получения всех записей из таблицы `cr`
    cursor.execute("SELECT Наименование, HTML_link, JSON_content FROM cr;")
    rows = cursor.fetchall()
    conn.close()

    all_final_chunks = []
    all_stats_before = []
    all_stats_after = []

    # Проходим по каждой строке (КР)
    for row in rows:
        diagnosis = row[0]
        url = row[1]
        json_content_str = row[2]

        # Парсим JSON-строку в словарь
        try:
            kr_data = json.loads(json_content_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for diagnosis '{diagnosis}': {e}")
            continue

        # --- Гибридное чанкирование ---
        enc = tiktoken.get_encoding("cl100k_base")
        stats_before = {
            "total_leaf_sections_processed": 0,
            "total_chunks_created_before_refinery": 0,
            "total_content_tokens_processed": 0,
            "chunk_lengths_tokens_before_refinery": [],
            "max_tokens_per_chunk_setting": max_tokens_per_content_chunk,
        }

        all_chunks_before_refinery = []
        collect_leaf_content_recursive(kr_data.get("sections", []), diagnosis, url, all_chunks_before_refinery, max_tokens_per_content_chunk, enc, parent_meta=None, stats=stats_before)

        # --- Применение OverlapRefinery ---
        temp_chunks_for_refinery = []
        metadata_mapping = []

        for idx, chunk_dict in enumerate(all_chunks_before_refinery):
            content_to_chunk = chunk_dict['content']
            large_chunk_size = 100000
            temp_sentence_chunks = SentenceChunker(chunk_size=large_chunk_size)(content_to_chunk)
            for sc_idx, sc_chunk_obj in enumerate(temp_sentence_chunks):
                temp_chunks_for_refinery.append(sc_chunk_obj)
                original_meta = {k: v for k, v in chunk_dict.items() if k not in ['content', 'token_count']}
                original_meta['original_chunk_index'] = idx
                original_meta['sub_chunk_index_in_original'] = sc_idx
                metadata_mapping.append(original_meta)

        # --- Применение Refinery ---
        approx_context_size = overlap_tokens / max_tokens_per_content_chunk if max_tokens_per_content_chunk > 0 else 0
        overlap_refinery = OverlapRefinery(context_size=approx_context_size, method="prefix", merge=True)
        refined_chunk_objects = overlap_refinery(temp_chunks_for_refinery)

        # --- Восстановление чанков с метаданными ---
        final_chunks = []
        stats_after = {
            "total_chunks_after_refinery": len(refined_chunk_objects),
            "chunk_lengths_tokens_after_refinery": [],
            "overlap_context_size_fraction_used": approx_context_size,
            "max_tokens_per_chunk_setting": max_tokens_per_content_chunk,
        }

        for idx, refined_chunk_obj in enumerate(refined_chunk_objects):
            refined_text = refined_chunk_obj.text
            refined_token_count = count_tokens_tiktoken(refined_text, enc)

            if idx < len(metadata_mapping):
                original_meta = metadata_mapping[idx].copy()
            else:
                print(f"Warning: Mismatch in metadata mapping at index {idx}. Using default metadata.")
                original_meta = {
                    "diagnosis": diagnosis, "url": url,
                    "parent_section_title": "", "parent_section_id": "", "parent_section_level": 0,
                    "leaf_title": "Unknown", "leaf_id": "Unknown", "leaf_level": 0,
                    "original_chunk_index": -1, "sub_chunk_index_in_original": -1
                }

            final_chunk_entry = {
                "diagnosis": original_meta.get("diagnosis", ""),
                "url": original_meta.get("url", ""),
                "parent_section_title": original_meta.get("parent_section_title", ""),
                "leaf_title": original_meta.get("leaf_title", ""),
                "parent_section_level": original_meta.get("parent_section_level", 0),
                "leaf_level": original_meta.get("leaf_level", 0),
                "original_chunk_index": original_meta.get("original_chunk_index", -1),
                "refined_chunk_index": idx,
                "chunk_text": refined_text,
                "token_count": refined_token_count,
                "path": f"{original_meta.get('parent_section_title', '')} > {original_meta.get('leaf_title', '')}" if original_meta.get('parent_section_title', '') else original_meta.get('leaf_title', '')
            }
            final_chunks.append(final_chunk_entry)
            stats_after["chunk_lengths_tokens_after_refinery"].append(refined_token_count)

        # --- Пересчет статистики ПОСЛЕ Refinery ---
        lengths_after = stats_after["chunk_lengths_tokens_after_refinery"]
        if lengths_after:
            stats_after["avg_chunk_length_tokens_after_refinery"] = round(sum(lengths_after) / len(lengths_after), 2)
            stats_after["min_chunk_length_tokens_after_refinery"] = min(lengths_after)
            stats_after["max_chunk_length_tokens_after_refinery"] = max(lengths_after)
        else:
            stats_after["avg_chunk_length_tokens_after_refinery"] = 0
            stats_after["min_chunk_length_tokens_after_refinery"] = 0
            stats_after["max_chunk_length_tokens_after_refinery"] = 0

        all_final_chunks.extend(final_chunks)
        all_stats_before.append(stats_before)
        all_stats_after.append(stats_after) # <-- Добавляем stats_after в список

    # --- Сохранение результатов ---
    df = pd.DataFrame(all_final_chunks)
    df.to_csv(output_csv_name, index=False, encoding='utf-8-sig')
    print(f"Чанки сохранены в CSV: {output_csv_name}")
    print(df.head())

    with open(output_json_name, 'w', encoding='utf-8') as f:
        json.dump(all_final_chunks, f, ensure_ascii=False, indent=2)
    print(f"Чанки сохранены в JSON: {output_json_name}")

    # --- Расчет ОБЩЕЙ статистики ---
    # Собираем статистику ДО Refinery
    total_stats_before = {
        "total_leaf_sections_processed": sum(s.get("total_leaf_sections_processed", 0) for s in all_stats_before),
        "total_chunks_created_before_refinery": sum(s.get("total_chunks_created_before_refinery", 0) for s in all_stats_before),
        "total_content_tokens_processed": sum(s.get("total_content_tokens_processed", 0) for s in all_stats_before),
        "chunk_lengths_tokens_before_refinery": [l for s in all_stats_before for l in s.get("chunk_lengths_tokens_before_refinery", [])],
        "max_tokens_per_chunk_setting": max_tokens_per_content_chunk,
    }

    # Собираем статистику ПОСЛЕ Refinery
    total_stats_after = {
        "total_chunks_after_refinery": sum(s.get("total_chunks_after_refinery", 0) for s in all_stats_after), # <-- Суммируем из all_stats_after
        "chunk_lengths_tokens_after_refinery": [l for s in all_stats_after for l in s.get("chunk_lengths_tokens_after_refinery", [])], # <-- Собираем длины из all_stats_after
        "overlap_context_size_fraction_used": sum(s.get("overlap_context_size_fraction_used", 0) for s in all_stats_after) / len(all_stats_after) if len(all_stats_after) > 0 else 0,
        "max_tokens_per_chunk_setting": max_tokens_per_content_chunk,
    }

    print("\n--- Общая статистика ДО Refinery ---")
    stats_before_calc = calculate_chunk_statistics(total_stats_before)
    # Переименовываем ключи для ясности
    renamed_stats_before = {k.replace("chunk_length_tokens", "chunk_length_tokens_before_refinery"): v for k, v in stats_before_calc.items()}
    for key, value in renamed_stats_before.items():
        print(f"{key}: {value}")

    print("\n--- Общая статистика ПОСЛЕ Refinery ---")
    stats_after_calc = calculate_chunk_statistics(total_stats_after) # <-- Вызов с total_stats_after
    # Переименовываем ключи для ясности
    renamed_stats_after = {k.replace("chunk_length_tokens", "chunk_length_tokens_after_refinery"): v for k, v in stats_after_calc.items()}
    for key, value in renamed_stats_after.items():
        print(f"{key}: {value}")


# --- 7. Использование ---
if __name__ == "__main__":
    # Параметры
    db_path = "db.sqlite"
    output_csv = "kr_structured_chunks_hybrid_overlap_30_v2.csv"
    output_json = "kr_structured_chunks_hybrid_overlap_30_v2.json"
    max_tokens_target = 384
    overlap_tokens_num = 30

    # Запуск
    process_kr_from_db(db_path, output_csv, output_json, max_tokens_target, overlap_tokens_num)
