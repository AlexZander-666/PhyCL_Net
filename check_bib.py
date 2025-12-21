import re

with open('arXiv/references.bib', 'r', encoding='utf-8') as f:
    content = f.read()

# 分析每个条目
entries = re.split(r'\n@', content)
print('=== BIB 文件分析 ===\n')

for entry in entries:
    if not entry.strip():
        continue
    
    # 提取 key
    key_match = re.search(r'^(\w+)\{(\w+)', entry)
    if not key_match:
        continue
    
    entry_type = key_match.group(1).lower()
    key = key_match.group(2)
    
    # 提取 title
    title_match = re.search(r'title\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
    title = title_match.group(1) if title_match else 'N/A'
    
    # 提取 journal/booktitle
    journal_match = re.search(r'journal\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
    booktitle_match = re.search(r'booktitle\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
    
    venue = None
    if journal_match:
        venue = journal_match.group(1)
    elif booktitle_match:
        venue = booktitle_match.group(1)
    
    # 提取 year
    year_match = re.search(r'year\s*=\s*\{?(\d{4})\}?', entry, re.IGNORECASE)
    year = year_match.group(1) if year_match else 'N/A'
    
    # 提取 doi
    doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', entry, re.IGNORECASE)
    has_doi = bool(doi_match)
    
    # 检查是否是预印本或缺少发表信息
    issue = None
    
    if 'arxiv' in (venue or '').lower():
        issue = 'arXiv preprint'
    elif 'preprint' in (venue or '').lower():
        issue = 'Preprint'
    elif not venue and entry_type in ['article', 'inproceedings']:
        issue = 'Missing venue'
    elif entry_type == 'inproceedings' and not has_doi and year >= '2020':
        # 近年会议论文没有 DOI 可能需要验证
        if 'ICLR' in (venue or ''):
            issue = 'ICLR conference - verify publication'
    
    if issue:
        print(f'{key} ({year})')
        print(f'  Type: {entry_type}')
        title_short = title[:60] + '...' if len(title) > 60 else title
        print(f'  Title: {title_short}')
        print(f'  Venue: {venue}')
        doi_str = 'Yes' if has_doi else 'No'
        print(f'  DOI: {doi_str}')
        print(f'  Issue: {issue}')
        print()
