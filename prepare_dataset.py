import os
import random
import shutil

# Organize data into train, valid, test dirs
os.chdir('Veriseti')

turler = ["Akıllı Anahtar", "Çift Taraflı İngiliz Anahtarı", "Çift Taraflı Yıldız Anahtar", "Cırcır Anahtarı", "Fort Pense", "Pense", "Tornavida", "Yankeski"]

if os.path.isdir('training/Pense/') is False: 
    os.mkdir('training')
    os.mkdir('validation')
    os.mkdir('test')

    for i in turler:
        shutil.move(f'{i}', 'training')
        os.mkdir(f'validation/{i}')
        os.mkdir(f'test/{i}')

        datasize = len(os.listdir(f"training/{i}"))

        valid_samples = random.sample(os.listdir(f'training/{i}'), int(datasize/10))
        for j in valid_samples:
            shutil.move(f'training/{i}/{j}', f'validation/{i}')

        test_samples = random.sample(os.listdir(f'training/{i}'), int(datasize/10))
        for k in test_samples:
            shutil.move(f'training/{i}/{k}', f'test/{i}')
os.chdir('../..')