import os
import re
import random
import sys

sys.path.append('../../')


def extract_title_from_tex(paper_path):
    if not os.path.exists(paper_path):
        return None
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    m = re.search(r'\\title\{([^}]+)\}', content)
    if m:
        return m.group(1)
    # Fallback: first non-empty line
    for line in content.splitlines():
        if line.strip():
            return line.strip()
    return None


def preserve_title_and_shuffle_body(text, shuffle_func):
    """Keep the first non-empty line (title) and apply shuffle_func to the rest."""
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
    shuffled_body = shuffle_func(body)
    return title + "\n" + shuffled_body


def shuffle_words(text):
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


def shuffle_sentences(text):
    # naive sentence split; keep punctuation
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p for p in parts if p.strip()]
    random.shuffle(parts)
    return ' '.join(parts)


def process_tex_file(src_path, dst_path, mode='sentences'):
    with open(src_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if mode == 'words':
        out = preserve_title_and_shuffle_body(content, shuffle_words)
    else:
        out = preserve_title_and_shuffle_body(content, shuffle_sentences)

    with open(dst_path, 'w', encoding='utf-8') as f:
        f.write(out)


def generate_shuffled_for_domain(domain, base_dir, modes=('sentences',), paraphrase_count=9):
    cleaned_dir = os.path.join(base_dir, 'cleaned')
    paraphrase_base = os.path.join(base_dir, 'paraphrased')

    src_tex = os.path.join(cleaned_dir, f'{domain}.tex')
    if not os.path.exists(src_tex):
        print(f"Source tex not found for {domain}: {src_tex}")
    else:
        for mode in modes:
            dst = os.path.join(cleaned_dir, f'{domain}_shuffle.tex')
            print(f"Writing shuffled {mode} source: {dst}")
            process_tex_file(src_tex, dst, mode=mode)

    # paraphrased: directories like paraphrased/<domain> / files 0.tex .. 8.tex
    domain_para_dir = os.path.join(paraphrase_base, domain)
    if not os.path.isdir(domain_para_dir):
        # also support numeric folder like '1_58'
        print(f"Paraphrase dir not found for {domain}: {domain_para_dir}")
        return

    for i in range(paraphrase_count):
        src_para = os.path.join(domain_para_dir, f'{i}.tex')
        if not os.path.exists(src_para):
            continue
        for mode in modes:
            dst_para = os.path.join(domain_para_dir, f'{i}_shuffle.tex')
            print(f"Writing shuffled {mode} paraphrase: {dst_para}")
            process_tex_file(src_para, dst_para, mode=mode)


def main():
    base_dir = '/Users/jlee0/Desktop/research/fine-tuning-or-retrieval/data/arxiv'
    cleaned_dir = os.path.join(base_dir, 'cleaned')
    domains = [f.replace('.tex','') for f in os.listdir(cleaned_dir) if f.endswith('.tex')]

    print(f"Found {len(domains)} domains to shuffle")
    for domain in domains:
        try:
            generate_shuffled_for_domain(domain, base_dir, modes=('sentences',), paraphrase_count=9)
        except Exception as e:
            print(f"Error processing {domain}: {e}")


if __name__ == '__main__':
    main()
