
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from lxml import etree


# ==========================================================
# БАЗОВЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================================


def safe_parse_xml(path: str | Path) -> etree._Element:
    """
    Безопасно парсит CDA-XML:

    - читает файл как байты;
    - отбрасывает весь мусор до первой декларации "<?xml";
    - игнорирует битые символы в UTF-8;
    - возвращает корневой элемент ClinicalDocument.
    """
    p = Path(path)
    with p.open("rb") as f:
        raw = f.read()

    text = raw.decode("utf-8", errors="ignore")
    start = text.find("<?xml")
    if start == -1:
        raise ValueError(f"XML declaration not found in file: {path}")

    cleaned = text[start:].strip()
    return etree.fromstring(cleaned.encode("utf-8"))


def _find_sections(root: etree._Element) -> List[etree._Element]:
    """
    Возвращает все <section> в документе (без учёта пространств имён).

    Не завязаны жёстко на структуру component/structuredBody:
    просто ищем все section в дереве.
    """
    return root.xpath(".//*[local-name()='section']")


def _text_content(node: Optional[etree._Element]) -> Optional[str]:
    """Собирает весь текст внутри узла. Пустые строки → None."""
    if node is None:
        return None
    txt = "".join(node.itertext()).strip()
    return txt or None


def extract_section_text(section: Optional[etree._Element]) -> Optional[str]:
    """
    Универсальный вытаскиватель текста из секции:
    - пробует взять первый <text>;
    - если его нет — весь текст секции.
    """
    if section is None:
        return None
    text_el = section.xpath(".//*[local-name()='text']")
    if text_el:
        return _text_content(text_el[0])
    return _text_content(section)


# ==========================================================
# ПАРСИНГ ОТДЕЛЬНЫХ СЕКЦИЙ
# ==========================================================


def extract_anamnesis_life(section: Optional[etree._Element]) -> Optional[List[Dict[str, Any]]]:
    """
    LANAM: собираем пары (название поля, значение).

    На выходе:
        [
            {"name": "...", "value": "..."},
            ...
        ]
    """
    if section is None:
        return None

    result: List[Dict[str, Any]] = []
    for obs in section.xpath(".//*[local-name()='observation']"):
        code_el = obs.xpath("./*[local-name()='code']")
        value_el = obs.xpath("./*[local-name()='value']")
        if not code_el or not value_el:
            continue

        display_name = code_el[0].get("displayName") or code_el[0].get("code")
        value_txt = _text_content(value_el[0])
        if value_txt:
            result.append({"name": display_name, "value": value_txt})

    return result or None


def extract_visit_data_from_ambsv(section: Optional[etree._Element]) -> Optional[Dict[str, Any]]:
    """
    AMBSV / AMBS: вытаскиваем ключевые вещи при визите:

    - complaints          (код 835 "Жалобы пациента")
    - objective_status    (код 836 "Объективные данные")
    - conclusion          (код 837 "Заключение")

    Если ни одно поле не найдено → возвращает None.
    """
    if section is None:
        return None

    visit: Dict[str, Any] = {
        "complaints": None,
        "objective_status": None,
        "conclusion": None,
    }

    for obs in section.xpath(".//*[local-name()='observation']"):
        code_el = obs.xpath("./*[local-name()='code']")
        if not code_el:
            continue

        code = code_el[0].get("code")
        value_el = obs.xpath("./*[local-name()='value']")
        if not value_el:
            continue

        value_txt = _text_content(value_el[0])
        if not value_txt:
            continue

        if code == "835":          # Жалобы пациента
            visit["complaints"] = value_txt
        elif code == "836":        # Объективные данные
            visit["objective_status"] = value_txt
        elif code == "837":        # Заключение
            visit["conclusion"] = value_txt

    return visit if any(visit.values()) else None


def extract_diagnosis_from_dgn(section: Optional[etree._Element]) -> Optional[List[Dict[str, Any]]]:
    """
    DGN: диагнозы.

    Возвращаем список словарей:
        [
            {
                "type": "...",          # Основное / сопутствующее и т.д.
                "code": "I67.4",
                "name": "Гипертензивная энцефалопатия",
                "character": "...",     # Характер заболевания (код 12014)
            },
            ...
        ]
    """
    if section is None:
        return None

    diags: List[Dict[str, Any]] = []

    for obs in section.xpath(".//*[local-name()='observation']"):
        relationships = obs.xpath("./*[local-name()='entryRelationship']/*[local-name()='observation']")
        if not relationships:
            continue

        diag: Dict[str, Any] = {
            "type": None,
            "code": None,
            "name": None,
            "character": None,
        }

        for rel_obs in relationships:
            code_el = rel_obs.xpath("./*[local-name()='code']")
            if not code_el:
                continue

            c = code_el[0]
            c_code = c.get("code")
            c_disp = c.get("displayName")
            value_nodes = rel_obs.xpath("./*[local-name()='value']")
            value_el = value_nodes[0] if value_nodes else None

            # Вид нозологической единицы (основное, сопутствующее и т.п.)
            if c_code == "1":
                diag["type"] = c_disp or "Основное заболевание"
                if value_el is not None:
                    diag["code"] = value_el.get("code")
                    diag["name"] = value_el.get("displayName") or _text_content(value_el)

            # Характер заболевания
            elif c_code == "12014":
                if value_el is not None:
                    diag["character"] = value_el.get("displayName") or _text_content(value_el)

        if any(diag.values()):
            diags.append(diag)

    return diags or None


def extract_reslab(section: Optional[etree._Element]) -> Optional[List[Dict[str, Any]]]:
    """
    RESLAB: результаты лабораторных исследований.

    Структура выхода:
        [
            {
                "test_name": "...",
                "value": "...",
                "unit": "...",
                "text": "...",          # строковое описание, если есть
                "ref_low": "...",
                "ref_high": "...",
            },
            ...
        ]

    Парсим достаточно универсально, чтобы работать и с 233, и с 235 шаблонами.
    """
    if section is None:
        return None

    results: List[Dict[str, Any]] = []

    # Берём все observation с кодом (включая вложенные)
    for obs in section.xpath(".//*[local-name()='observation' and ./*[local-name()='code']]"):
        code_el = obs.xpath("./*[local-name()='code']")[0]

        # Имя теста: displayName / code / originalText / fallback
        test_name = (
            code_el.get("displayName")
            or code_el.get("code")
        )

        # Текстовое описание, если есть
        text_el = obs.xpath("./*[local-name()='text']")
        text_val = _text_content(text_el[0]) if text_el else None

        # Значение и единицы: стараемся достать из <value> у этого observation
        value_nodes = obs.xpath("./*[local-name()='value']")
        value_el = value_nodes[0] if value_nodes else None

        value: Optional[str] = None
        unit: Optional[str] = None
        if value_el is not None:
            value = value_el.get("value") or _text_content(value_el)
            unit = value_el.get("unit")

        # referenceRange: пробуем вытащить low/high, если они есть
        ref_low: Optional[str] = None
        ref_high: Optional[str] = None
        for ref in obs.xpath(".//*[local-name()='referenceRange']//*[local-name()='observationRange']"):
            low = ref.xpath(".//*[local-name()='low']")
            high = ref.xpath(".//*[local-name()='high']")
            if low:
                ref_low = low[0].get("value")
            if high:
                ref_high = high[0].get("value")

        # Если ничего осмысленного нет — пропускаем
        if not any([test_name, value, text_val]):
            continue

        results.append(
            {
                "test_name": test_name,
                "value": value,
                "unit": unit,
                "text": text_val,
                "ref_low": ref_low,
                "ref_high": ref_high,
            }
        )

    return results or None


def extract_services(section: Optional[etree._Element]) -> Optional[List[Dict[str, Any]]]:
    """
    SERVICES: оказанные услуги.

    На выходе:
        [
            {"code": "...", "name": "...", "date": "YYYYMMDD..."},
            ...
        ]
    """
    if section is None:
        return None

    services: List[Dict[str, Any]] = []

    for obs in section.xpath(".//*[local-name()='observation']"):
        code_el = obs.xpath("./*[local-name()='code']")
        if not code_el:
            continue

        c = code_el[0]
        code = c.get("code")
        name = c.get("displayName") or code

        eff_el = obs.xpath("./*[local-name()='effectiveTime']")
        date = eff_el[0].get("value") if eff_el else None

        services.append({"code": code, "name": name, "date": date})

    return services or None


# ==========================================================
# ОБЩИЙ ПАРСЕР ДОКУМЕНТА
# ==========================================================


def parse_ru_cda(path: str | Path) -> Dict[str, Any]:
    """
    Главная точка входа.

    На вход:
        path: путь к CDA-XML файлу.

    На выход:
        {
            "diagnosis":        [...],
            "anamnesis_life":  [...],
            "ambs":             None (зарезервировано под доп. парсинг),
            "visit_data":       {...},
            "services":         [...],
            "lab_results":      [...],
            "state_admission":  "...",   # 235: STATEADM
            "state_discharge":  "...",   # 235: STATEDIS
        }
    """
    root = safe_parse_xml(path)
    sections = _find_sections(root)

    dgn = lanam = ambs = ambsv = services = reslab = None
    stateadm = statedis = None

    for s in sections:
        code_el = s.xpath("./*[local-name()='code']")
        if not code_el:
            continue

        c_code = code_el[0].get("code")
        if c_code == "DGN":
            dgn = s
        elif c_code == "LANAM":
            lanam = s
        elif c_code == "AMBS":
            ambs = s
        elif c_code == "AMBSV":
            ambsv = s
        elif c_code == "SERVICES":
            services = s
        elif c_code == "RESLAB":
            reslab = s
        elif c_code == "STATEADM":
            stateadm = s
        elif c_code == "STATEDIS":
            statedis = s

    visit_data = extract_visit_data_from_ambsv(ambsv) or extract_visit_data_from_ambsv(ambs)

    return {
        "diagnosis":       extract_diagnosis_from_dgn(dgn),
        "anamnesis_life":  extract_anamnesis_life(lanam),
        "ambs":            None,
        "visit_data":      visit_data,
        "services":        extract_services(services),
        "lab_results":     extract_reslab(reslab),
        "state_admission": extract_section_text(stateadm),
        "state_discharge": extract_section_text(statedis),
    }


# ==========================================================
# ПРОСТОЙ "XML VIEWER"
# ==========================================================


def quick_xml_overview(path: str | Path, max_sections: int = 10) -> List[Dict[str, Optional[str]]]:
    """
    Простой просмотрщик: показывает список секций и кусок текста внутри.

    Возвращает список:
        [
            {"code": "DGN", "name": "Диагнозы", "preview": "..."},
            ...
        ]
    """
    root = safe_parse_xml(path)
    sections = _find_sections(root)

    overview: List[Dict[str, Optional[str]]] = []

    for s in sections[:max_sections]:
        code_el = s.xpath("./*[local-name()='code']")
        sec_code = code_el[0].get("code") if code_el else None
        disp = code_el[0].get("displayName") if code_el else None

        text_el = s.xpath(".//*[local-name()='text']") or s.xpath(".//*[local-name()='value']")
        txt = _text_content(text_el[0]) if text_el else ""

        if txt and len(txt) > 200:
            txt = txt[:197] + "..."

        overview.append(
            {
                "code": sec_code,
                "name": disp,
                "preview": txt or None,
            }
        )

    return overview


# ==========================================================
# CLI-ТЕСТ (можно запускать как скрипт)
# ==========================================================


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Парсер HL7 CDA (RU) + быстрый XML viewer.")
    parser.add_argument("path", help="Путь к CDA XML файлу")
    parser.add_argument("--overview", action="store_true", help="Показать только обзор секций (XML viewer)")
    args = parser.parse_args()

    if args.overview:
        for sec in quick_xml_overview(args.path, max_sections=50):
            print(f"{sec['code']:12} | {sec['name'] or '-'}")
            if sec["preview"]:
                print("   ", sec["preview"])
            print()
    else:
        data = parse_ru_cda(args.path)
        print(json.dumps(data, ensure_ascii=False, indent=2))
