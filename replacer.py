import os
import json
import re
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

SOURCE_DIR = "./docs/src/"
OUTPUT_DIR = "./docs/knowledge_base/"
MAPPING_FILE = "./docs/terms_map.json"


def replacement_word_form(name_to_change, new_base_name):

    parsed_old = morph.parse(name_to_change)
    if not parsed_old:
        return new_base_name

    is_possessive = any('Poss' in p.tag for p in parsed_old)

    best_parse = None
    for p in parsed_old:
        if 'Name' in p.tag or 'Surn' in p.tag or 'Poss' in p.tag:
            best_parse = p
            break
    if not best_parse:
        best_parse = parsed_old[0]

    grammemes = best_parse.tag.grammemes
    required_tags = {g for g in grammemes if g in {
        'sing', 'plur', 'nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'masc', 'femn', 'neut'
    }}

    if is_possessive:
        if new_base_name.endswith(('я', 'а')):
            base_poss = new_base_name[:-1] + "ин"
        else:
            base_poss = new_base_name + "ин"
        parsed_new = morph.parse(base_poss)
    else:
        parsed_new = morph.parse(new_base_name)

    if not parsed_new:
        return new_base_name

    inflected = parsed_new[0].inflect(required_tags)

    if inflected:
        result = inflected.word
        if name_to_change.isupper():
            result = result.upper()
        elif name_to_change[0].isupper():
            result = result.capitalize()
        return result
    return new_base_name


def process_text(text, sorted_replacements):
    for item in sorted_replacements:
        old_name = item["key"]
        new_name = item["value"]

        parsed_old_mapping = morph.parse(old_name)
        if not parsed_old_mapping:
            continue

        old_lemma = parsed_old_mapping[0].normal_form
        words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text)
        unique_words = set(words)

        for word in unique_words:
            parsed = morph.parse(word)

            is_match = False
            for p in parsed:
                if p.normal_form == old_lemma:
                    is_match = True
                    break
                if 'Poss' in p.tag and p.normal_form.startswith(old_lemma[:-1].lower()):
                    is_match = True
                    break

            if is_match:
                replacement_form = replacement_word_form(word, new_name)
                text = re.sub(rf'\b{word}\b', replacement_form, text)
    return text


def replace_names(src_dir, dest_dir, mapping_file_path):
    with open(mapping_file_path, "r", encoding="utf-8") as f:
        replacement_mapping = json.load(f)

    sorted_replacements = sorted(
        replacement_mapping["replacement"],
        key=lambda x: len(x["key"]),
        reverse=True
    )

    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = process_text(content, sorted_replacements)
            new_file_path = os.path.join(dest_dir, file)

            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"[saved]: {new_file_path}")


if __name__ == "__main__":
    replace_names(SOURCE_DIR, OUTPUT_DIR, MAPPING_FILE)
    print("success!")
