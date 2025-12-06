import json
from bs4 import BeautifulSoup
import re

def extract_diagnosis_from_soup(soup: BeautifulSoup) -> str:
    # 1) Ищем явный заголовок болезни: <span class="title_content">
    span = soup.find("span", class_="title_content")
    if span and span.get_text(strip=True):
        return span.get_text(strip=True)
    # 2) Ищем блок block_title_name
    blk = soup.find("div", class_=re.compile(r"block_title_name"))
    if blk:
        t = blk.get_text("\n", strip=True)
        if t:
            return t.splitlines()[-1]
    # 3) fallback: первый <p id="doc_...">
    p = soup.find("p", id=re.compile(r"^doc_"))
    if p:
        return p.get_text(strip=True)
    # 4) последний резерв - h1/h2
    h = soup.find(["h1","h2"])
    if h:
        return h.get_text(strip=True)
    return "Неизвестно"


def classify_category(section_title: str) -> str:
    title_lower = section_title.lower()
    if any(kw in title_lower for kw in ["симптом", "клиническая картина", "жалобы"]):
        return "Симптомы"
    elif any(kw in title_lower for kw in ["лечение", "терапия", "медикаментозная", "хирургическое"]):
        return "Лечение"
    elif any(kw in title_lower for kw in ["диагностика", "инструментальные", "лабораторные"]):
        return "Диагностика"
    elif "реабилитация" in title_lower:
        return "Реабилитация"
    elif "профилактика" in title_lower:
        return "Профилактика"
    elif any(kw in title_lower for kw in ["определение", "этиология", "патогенез", "эпидемиология"]):
        return "Общая информация"
    else:
        return "Прочее"

def parse_kr_structure(soup: BeautifulSoup) -> list:
    all_tags = soup.find_all(True)

    def is_title_tag(tag):
        if not hasattr(tag, "name"):
            return False
        tid = tag.get("id","")
        if tag.name == "p" and tid and re.match(r"^(doc_|doc_crat_info_|doc_diag_)", tid):
            txt = tag.get_text(strip=True)
            if re.match(r"^\d+(\.\d+)*\b", txt):
                return True
            return True
        if tag.name in ("h1","h2","h3"):
            txt = tag.get_text(strip=False) # strip=True
            if re.match(r"^\d+(\.\d+)*\b", txt):
                return True
        return False

    def get_level_from_title_text(text, tag):
        m = re.match(r"^\s*(\d+(?:\.\d+)*)", text)
        if m:
            nums = m.group(1).split(".")
            return len(nums)
        tid = tag.get("id","")
        nums = re.findall(r"\d+", tid)
        if nums:
            return len(nums)
        return 1

    # находим индексы заголовков в порядке документа
    title_indices = [i for i,t in enumerate(all_tags) if is_title_tag(t)]

    sections_flat = []
    for k,pos in enumerate(title_indices):
        tag = all_tags[pos]
        title_text = tag.get_text(" ", strip=True)
        level = get_level_from_title_text(title_text, tag)
        next_idx = title_indices[k+1] if k+1 < len(title_indices) else len(all_tags)
        content_parts = []
        for j in range(pos+1, next_idx):
            t2 = all_tags[j]
            if t2.name in ("script","style"):
                continue
            txt = t2.get_text(" ", strip=False) # strip=True
            if txt:
                content_parts.append(txt)
        content_text = "\n".join(content_parts) # .strip()

        # --- Удаляем все дублирующиеся заголовки в начале контента ---
        normalized_title = re.sub(r"\s+", " ", title_text).strip()

        lines = content_text.splitlines()
        while lines and re.sub(r"\s+", " ", lines[0]).strip() == normalized_title:
            lines.pop(0)
        content_text = "\n".join(lines) # .strip()

        sections_flat.append({
            "title": title_text,
            "id": tag.get("id",""),
            "level": level,
            "content": content_text,
            "tag_name": tag.name,
            "subsections": []
        })

    # строим дерево по уровням (стек)
    root = []
    stack = []
    for sec in sections_flat:
        node = {k: sec[k] for k in ("title","id","level","content","tag_name")}
        node["subsections"] = []
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if not stack:
            root.append(node)
            stack.append(node)
        else:
            stack[-1]["subsections"].append(node)
            stack.append(node)
    return root



def kr_to_json(html: str, url = "none") -> dict:
    """Основная функция: HTML-файл → структурированный JSON"""
#    with open(html_file_path, "r", encoding="utf-8") as f:
#        html = f.read()

    soup = BeautifulSoup(html, "lxml")

    diagnosis = extract_diagnosis_from_soup(soup)
    sections = parse_kr_structure(soup)

    return {
        "sections": sections
    }

# --- Использование ---
if __name__ == "__main__":
    html_file = "page.html"  # Файл, который создал скраппер
    url = "https://cr.minzdrav.gov.ru/view-cr/946_1"  # URL КР

    kr_data = kr_to_json(html_file, url)

    # Сохраняем в файл
    output_file = "kr_structured.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kr_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Успешно! Диагноз: {kr_data['diagnosis']}")
    print(f"📄 Найдено {len(kr_data['sections'])} разделов")
    print(f"💾 Сохранено в: {output_file}")