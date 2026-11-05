import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET
import zipfile

ROOT = pathlib.Path('/scratch/wz55191/Github/prompt-injection-as-role-confusion')
REPORT = ROOT / 'experiments/autodan-role-probe/outputs/all-tested-template-summary-20261105'
BOOK = ROOT / 'outputs/01a1a996-09a6-7c50-a32c-9100dfa738f7/all-tested-template-summary.xlsx'
data = json.loads((REPORT / 'reporting_data.json').read_text())
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
sha = lambda b: hashlib.sha256(b).hexdigest()

with zipfile.ZipFile(BOOK) as archive:
    shared = []
    if 'xl/sharedStrings.xml' in archive.namelist():
        for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns):
            shared.append(''.join(t.text or '' for t in item.findall('.//s:t', ns)))
    workbook = ET.fromstring(archive.read('xl/workbook.xml'))
    sheet_names = [node.attrib['name'] for node in workbook.findall('s:sheets/s:sheet', ns)]
    assert sheet_names == ['Summary', 'Runs', 'ProbeCases', 'Templates']
    sheets = {}
    cells = {}
    for i, name in enumerate(sheet_names, 1):
        sheet = ET.fromstring(archive.read(f'xl/worksheets/sheet{i}.xml'))
        sheets[name] = sheet
        values = {}
        for cell in sheet.findall('.//s:sheetData/s:row/s:c', ns):
            value = cell.find('s:v', ns)
            typ = cell.attrib.get('t')
            if typ == 's':
                parsed = shared[int(value.text)] if value is not None else ''
            elif typ == 'inlineStr':
                parsed = ''.join(t.text or '' for t in cell.findall('.//s:t', ns))
            elif typ == 'str':
                parsed = value.text if value is not None else ''
            elif typ == 'b':
                parsed = value is not None and value.text == '1'
            elif typ == 'e':
                raise AssertionError((name, cell.attrib['r'], value.text))
            else:
                parsed = float(value.text) if value is not None and value.text else None
            values[cell.attrib['r']] = parsed
        cells[name] = values
        pane = sheet.find('s:sheetViews/s:sheetView/s:pane', ns)
        assert pane is not None
        assert pane.attrib == {'xSplit': '2', 'ySplit': '8', 'topLeftCell': 'C9', 'activePane': 'bottomRight', 'state': 'frozen'}
        view = sheet.find('s:sheetViews/s:sheetView', ns)
        assert view.attrib.get('showGridLines') == '0'

    table_features = []
    for name in sorted(n for n in archive.namelist() if n.startswith('xl/tables/') and n.endswith('.xml')):
        table = ET.fromstring(archive.read(name))
        auto_filter = table.find('s:autoFilter', ns)
        assert auto_filter is not None, name
        assert auto_filter.attrib['ref'] == table.attrib['ref']
        table_features.append({'name': table.attrib['name'], 'range': table.attrib['ref'], 'auto_filter': True})
    assert len(table_features) == 5

    styles = ET.fromstring(archive.read('xl/styles.xml'))
    formats = {int(item.attrib['numFmtId']): item.attrib['formatCode'] for item in styles.findall('s:numFmts/s:numFmt', ns)}
    xfs = styles.findall('s:cellXfs/s:xf', ns)
    def cell_format(sheet_name, address):
        cell = sheets[sheet_name].find(f'.//s:c[@r="{address}"]', ns)
        return formats.get(int(xfs[int(cell.attrib.get('s', '0'))].attrib['numFmtId']))

    all_role_values = 0
    for row_index, source in enumerate(data['summary_rows'], 9):
        for j, role in enumerate(data['role_order']):
            addr = f'{chr(70 + j)}{row_index}'
            assert cells['Summary'][addr] == source['role_probabilities'][role]
            assert cell_format('Summary', addr) == '0.000000E+00'
            all_role_values += 1
        assert cells['Summary'][f'L{row_index}'] == source['asr']['asr_lower_bound']
        assert cells['Summary'][f'M{row_index}'] == source['asr']['asr_upper_bound']
        assert cells['Summary'][f'N{row_index}'] == source['asr']['scored_asr']
    for row_index, source in enumerate(data['probe_cases'], 9):
        for j, role in enumerate(data['role_order']):
            addr = f'{chr(70 + j)}{row_index}'
            assert cells['ProbeCases'][addr] == source['role_probabilities'][role]
            assert cell_format('ProbeCases', addr) == '0.000000E+00'
            all_role_values += 1

    endings = {'LF': '\n', 'CRLF': '\r\n', 'CR': '\r', 'EOF': ''}
    reconstructed = {}
    line_counts = {}
    for row_index, source in enumerate(data['template_lines'], 9):
        template_id = cells['Templates'][f'A{row_index}']
        line_no = int(cells['Templates'][f'B{row_index}'])
        text = cells['Templates'].get(f'C{row_index}') or ''
        ending = cells['Templates'][f'D{row_index}']
        assert (template_id, line_no, text, ending) == (source['template_id'], source['line_number'], source['text'], source['line_ending_label'])
        assert endings[ending] == source['line_ending']
        line_counts[template_id] = line_counts.get(template_id, 0) + 1
        reconstructed[template_id] = reconstructed.get(template_id, '') + text + endings[ending]
    reconstructed_shas = {}
    for source in data['summary_rows']:
        template_id = source['template_id']
        reconstructed_shas[template_id] = sha(reconstructed[template_id].encode('utf-8'))
        assert reconstructed_shas[template_id] == source['template_raw_sha256']
        assert reconstructed_shas[template_id] == sha(pathlib.Path(source['template_file']).read_bytes())

report = {
    'passed': True,
    'workbook': str(BOOK),
    'workbook_sha256': sha(BOOK.read_bytes()),
    'reporting_data_sha256': sha((REPORT / 'reporting_data.json').read_bytes()),
    'sheet_names': sheet_names,
    'native_filters': table_features,
    'native_freeze': {'sheets': 4, 'rows': 8, 'columns': 2, 'top_left_cell': 'C9'},
    'gridlines_hidden': True,
    'role_values_and_scientific_number_formats_verified': all_role_values,
    'role_number_format': '0.000000E+00',
    'raw_text_rows_verified': len(data['template_lines']),
    'template_line_counts': line_counts,
    'reconstructed_template_sha256': reconstructed_shas,
}
(REPORT / 'workbook_qa/native_xml_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))
