import csv
from email import header
from unicodedata import name
import pandas as pd
#---------------------- DRISHTI---------------------

# path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/DRISHTI/Drishti-GS1_diagnosis.xlsx'

# diagnosis = pd.read_excel(path,header=2,usecols=[1,2,8])
# print(diagnosis)
# row,col= diagnosis.shape
# dst_filename = '/mnt/Almacenamiento/ODOC_segmentation/data/images/DRISHTI/labels.csv'
# with open(dst_filename, 'a+') as tags:
#     writer = csv.writer(tags)
#     print(tags)
#     writer.writerow(['Name','label'])
# for i in diagnosis.index: 
#     with open(dst_filename, 'a+') as tags:
#         writer = csv.writer(tags)
#         name = diagnosis['Unnamed: 1'][i]
#         name = str(name).replace("'",'')
#         name = name.replace('drishtiGS_','')
#         #print(name)
#         diag = diagnosis['Unnamed: 8'][i]
#         if diag == 'Normal':
#             writer.writerow([name,'Non-glaucomatous'])
#         else:
#             writer.writerow([name,diag])

#------------------- REFUGE ---------------------------------------

# path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/REFUGE/test_dataset/GT/GT.xlsx'
# dst_filename = '/mnt/Almacenamiento/ODOC_segmentation/data/images/REFUGE/labels.csv'

# diagnosis = pd.read_excel(path)
# print(diagnosis)
# count_name = 801

# with open(dst_filename, 'a+') as tags:
#     writer = csv.writer(tags)
#     print(tags)
#     writer.writerow(['Name','label'])

# for i in diagnosis.index: 
#     with open(dst_filename, 'a+') as tags:
#         writer = csv.writer(tags)

#         diag = diagnosis['Glaucoma(1) Label'][i]
#         name = f"{count_name:04}"
#         print(name)
#         if diag == 0:
#             writer.writerow([name,'Non-glaucomatous'])
#         elif diag == 1:
#             writer.writerow([name,'Glaucomatous'])

#     count_name += 1

#--------------------ORIGA -----------------------------------

path = '/mnt/Almacenamiento/ODOC_segmentation/raw_data/ORIGA/labels.xlsx'
dst_filename = '/mnt/Almacenamiento/ODOC_segmentation/data/images/ORIGA/labels.csv'

diagnosis = pd.read_excel(path)
print(diagnosis)

with open(dst_filename, 'a+') as tags:
    writer = csv.writer(tags)
    print(tags)
    writer.writerow(['Name','label'])

for i in diagnosis.index: 
    with open(dst_filename, 'a+') as tags:
        writer = csv.writer(tags)
        name = diagnosis['filename'][i]
        name = name.replace('.jpg','')
        name = name.replace("'","")
        diag = diagnosis['diagnosis(glaucoma=True)'][i]
        if diag == False:
            writer.writerow([name,'Non-glaucomatous'])
        elif diag == True:
            writer.writerow([name, 'Glaucomatous'])
        else:
            print('No entro')
