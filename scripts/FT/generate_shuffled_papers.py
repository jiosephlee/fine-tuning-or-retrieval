import os
import re
import random
import sys

sys.path.append('../../')


def preserve_title_and_apply(text, fn):
    """Keep the first non-empty line as the title and apply fn to the rest."""
    lines = text.split('\n')
    title_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            title_idx = i
            break
    if title_idx is None:
        return text
    title = lines[title_idx]
    body = '\n'.join(lines[title_idx+1:])
    return title + '\n' + fn(body)


def shuffle_words(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def shuffle_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p for p in parts if p.strip()]
    random.shuffle(parts)
    return ' '.join(parts)


def process_and_write(src_path, dst_path, mode='sentences'):
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if mode == 'words':
        out = preserve_title_and_apply(content, shuffle_words)
    else:
        out = preserve_title_and_apply(content, shuffle_sentences)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(out)


def generate_shuffled_for_domain(domain, base_dir, paraphrase_count=9):
    cleaned_dir = os.path.join(base_dir, 'cleaned')
    paraphrase_base = os.path.join(base_dir, 'paraphrased')
    shuffled_root = os.path.join(base_dir, 'shuffled')

    # prepare shuffled output dirs
    os.makedirs(os.path.join(shuffled_root, 'cleaned'), exist_ok=True)

    src_tex = os.path.join(cleaned_dir, f'{domain}.tex')
    if os.path.exists(src_tex):
        if 'shuffle' in os.path.basename(src_tex):
            print(f"Skipping source (already shuffle-like): {src_tex}")
        else:
            dst_words = os.path.join(shuffled_root, 'cleaned', f'{domain}_shuffle_words.tex')
            dst_sent = os.path.join(shuffled_root, 'cleaned', f'{domain}_shuffle_sentences.tex')
            print(f"Shuffling source {domain} -> {os.path.basename(dst_words)}, {os.path.basename(dst_sent)}")
            process_and_write(src_tex, dst_words, mode='words')
            process_and_write(src_tex, dst_sent, mode='sentences')
    else:
        print(f"Source tex not found for {domain}: {src_tex}")

    domain_para_dir = os.path.join(paraphrase_base, domain)
    out_para_dir = os.path.join(shuffled_root, 'paraphrased', domain)
    os.makedirs(out_para_dir, exist_ok=True)

    if not os.path.isdir(domain_para_dir):
        print(f"Paraphrase dir not found for {domain}: {domain_para_dir}")
        return

    for i in range(paraphrase_count):
        src_para = os.path.join(domain_para_dir, f'{i}.tex')
        if not os.path.exists(src_para):
            continue
        if 'shuffle' in os.path.basename(src_para):
            print(f"Skipping paraphrase (already shuffle-like): {src_para}")
            continue
        dst_words = os.path.join(out_para_dir, f'{i}_shuffle_words.tex')
        dst_sent = os.path.join(out_para_dir, f'{i}_shuffle_sentences.tex')
        print(f"Shuffling paraphrase {domain}/{i} -> {os.path.basename(dst_words)}, {os.path.basename(dst_sent)}")
        process_and_write(src_para, dst_words, mode='words')
        process_and_write(src_para, dst_sent, mode='sentences')


def main():
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv'
    cleaned_dir = os.path.join(base_dir, 'cleaned')
    # exclude files that already contain 'shuffle' in their names
    domains = [f.replace('.tex','') for f in os.listdir(cleaned_dir) if f.endswith('.tex') and 'shuffle' not in f]

    print(f"Found {len(domains)} domains to shuffle")
    for domain in domains:
        try:
            generate_shuffled_for_domain(domain, base_dir, paraphrase_count=9)
        except Exception as e:
            print(f"Error processing {domain}: {e}")


if __name__ == '__main__':
    main()
