import os
import xml.etree.ElementTree as ET
from collections import Counter

# Caminho para o diretório com os arquivos XML
directory_path = '/home/mstveras/ssd-360/dataset/train/labels'

# Dicionário para contar as instâncias de cada nome
name_counter = Counter()

# Iterar sobre cada arquivo no diretório
for filename in os.listdir(directory_path):
    if filename.endswith('.xml'):
        # Construir o caminho completo para o arquivo
        file_path = os.path.join(directory_path, filename)
        
        # Parsear o arquivo XML
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extrair os valores de <name> dentro de cada <object>
        for obj in root.findall('.//object/name'):
            name = obj.text
            if name:
                name_counter[name] += 1

# Exibir os resultados
for name, count in name_counter.items():
    print(f"Name: {name}, Instances: {count}")
