"""Resolve the unverified bibliography entries via the public Crossref API."""
import json, urllib.parse, urllib.request

QUERIES = {
    'hnse2025': 'Machine learning-based prediction of power demand and fuel '
                'consumption of a power plant: A case study from Bangladesh',
    'ecai2025': 'Comparative Analysis of Deep Learning Models for Long-Term '
                'Electricity Demand Forecasting in Bangladesh Using Web-Scraped Data',
    'coda_china2020': "Compositional data techniques for forecasting dynamic change "
                      "in China's energy consumption structure",
    'coda_mix2025': 'Compositional regression analysis of the energy mix and its determinants',
    'dataset_descriptor': 'Multi-year dataset on daily electricity demand, generation, '
                          'load shedding, and external conditions in Bangladesh',
}
UA = {'User-Agent': 'research-script/1.0 (mailto:wshuvo360@gmail.com)'}

for key, title in QUERIES.items():
    url = ('https://api.crossref.org/works?rows=3&select=title,author,container-title,'
           'volume,issue,page,issued,DOI,type&query.bibliographic='
           + urllib.parse.quote(title))
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            items = json.load(r)['message']['items']
    except Exception as e:
        print(f'{key}: FETCH FAILED ({e})'); continue

    print(f'\n=== {key} ===')
    for it in items[:2]:
        t = (it.get('title') or ['?'])[0]
        # only accept a close title match
        overlap = len(set(t.lower().split()) & set(title.lower().split()))
        auth = it.get('author', [])
        names = '; '.join(
            f"{a.get('given','?')} {a.get('family','?')}" for a in auth) or '(none listed)'
        yr = (it.get('issued', {}).get('date-parts') or [[None]])[0][0]
        print(f'  match({overlap} words): {t[:95]}')
        print(f'    authors : {names[:200]}')
        print(f'    journal : {(it.get("container-title") or ["?"])[0]}')
        print(f'    vol/iss/pages/year: {it.get("volume","-")}/{it.get("issue","-")}/'
              f'{it.get("page","-")}/{yr}')
        print(f'    doi     : {it.get("DOI","-")}   type: {it.get("type","-")}')
