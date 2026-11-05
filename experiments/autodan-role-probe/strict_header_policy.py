"""Read-only selection diagnostics for repeated headers, lines and paragraphs.

This sidecar does not modify the frozen search admission policy or any text.
It adds the user's final short-template selection restriction to old scores.
"""
from collections import Counter
import re
import unicodedata

from diverse_policy import _MARKUP

ROLES='system|developer|user|assistant|tool|browser|python|human'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC',text).casefold().split())


def strict_header_metrics(template):
    headers=[]
    markers=list(_MARKUP.finditer(template))
    def name(marker):
        return ''.join(c for c in unicodedata.normalize('NFKC',marker.group()).casefold()if c.isalpha())
    names=[name(marker)for marker in markers]
    for i,(marker,kind)in enumerate(zip(markers,names)):
        if kind in ('start','imstart','begin'):
            for j in range(i+1,min(len(markers),i+8)):
                if names[j]in ('start','imstart','begin','end','imend','call','return'):
                    break
                if names[j]in ('message','imsep','content'):
                    value=template[marker.start():markers[j].end()]
                    headers.append(('near_message_header',normalized(value)))
                    break
        raw=marker.group()
        if not raw.lstrip().startswith('</')and re.fullmatch(ROLES,kind):
            headers.append(('xml_role_header',normalized(raw)))
    # Plain role labels are recognized only at line starts, so body mentions of
    # a role do not become artificial duplicate headers.
    role_lines=re.compile(r'(?m)^[ \t]*(?:#{1,6}[ \t]*)?('+ROLES+r')[ \t]*(?::|(?=\r?$))',re.I)
    for match in role_lines.finditer(template):
        headers.append(('plain_role_header',match.group(1).casefold()))
    json_roles=re.compile(r'''["']role["']\s*:\s*["']('''+ROLES+r''')["']''',re.I)
    for match in json_roles.finditer(template):
        headers.append(('json_role_header',match.group(1).casefold()))
    header_counts=Counter(headers)
    lines=[normalized(x)for x in template.replace('{goal}','').splitlines()if normalized(x)]
    paragraphs=[normalized(x)for x in re.split(r'\n[ \t\r]*\n',template.replace('{goal}',''))if normalized(x)]
    line_counts,paragraph_counts=Counter(lines),Counter(paragraphs)
    duplicate_headers=[dict(kind=key[0],header=key[1],copies=count)for key,count in header_counts.items()if count>1]
    duplicate_lines=[dict(text=key,copies=count)for key,count in line_counts.items()if count>1]
    duplicate_paragraphs=[dict(text=key,copies=count)for key,count in paragraph_counts.items()if count>1]
    return dict(allowed=not(duplicate_headers or duplicate_lines or duplicate_paragraphs),
      exact_duplicate_headers=duplicate_headers,exact_duplicate_lines=duplicate_lines,
      exact_duplicate_paragraphs=duplicate_paragraphs,header_count=len(headers),
      maximum_identical_header_copies=max(header_counts.values(),default=0),
      maximum_identical_line_copies=max(line_counts.values(),default=0),
      maximum_identical_paragraph_copies=max(paragraph_counts.values(),default=0),
      text_modified=False,prediction_or_fitness=False,
      normalization_for_diagnostics_only='NFC/casefold/collapse whitespace; candidate bytes unchanged')
