import csv
import json

# CSV ve JSON dosya adlarını belirt
csv_dosyasi = 'framingham.csv'
json_dosyasi = 'framingham.json'

# CSV'yi oku ve JSON'a yaz
with open(csv_dosyasi, mode='r', encoding='utf-8') as csv_file:
    csv_okuyucu = csv.DictReader(csv_file)
    veri_listesi = list(csv_okuyucu)

with open(json_dosyasi, mode='w', encoding='utf-8') as json_file:
    json.dump(veri_listesi, json_file, indent=4, ensure_ascii=False)
