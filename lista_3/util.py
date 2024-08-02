import os
import shutil

def mover_arquivos_para_raiz(pasta_raiz):
    # Percorre todas as subpastas dentro da pasta raiz
    for subdir, _, files in os.walk(pasta_raiz):
        # Ignora a pasta raiz
        if subdir == pasta_raiz:
            continue
        
        # Move todos os arquivos das subpastas para a raiz
        for file in files:
            caminho_origem = os.path.join(subdir, file)
            caminho_destino = os.path.join(pasta_raiz, file)
            shutil.move(caminho_origem, caminho_destino)
            print(f'Movido: {caminho_origem} -> {caminho_destino}')
        
        # Após mover os arquivos, remove a subpasta se estiver vazia
        if not os.listdir(subdir):
            os.rmdir(subdir)
            print(f'Subpasta removida: {subdir}')

# Especifique o caminho para a pasta raiz
pasta_raiz = 'lista_3/archive'

mover_arquivos_para_raiz(pasta_raiz)
