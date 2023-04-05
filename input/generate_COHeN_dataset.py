#!/bin/bash/python

import re
import math
import warnings
import pandas as pd
from datasets import load_dataset, DatasetDict, ClassLabel

warnings.filterwarnings("ignore")

df = pd.read_csv('Tanakh_data.csv', index_col = 'Verse')

ABH = pd.concat([df['Gen 49:2':'Gen 49:27'], 
                 df['Exod 15:1':'Exod 15:18'], 
                 df['Num 23:7' : 'Num 23:10'], 
                 df['Num 23:18' : 'Num 23:24'], 
                 df['Num 24:3' : 'Num 24:9'], 
                 df['Num 24:15' : 'Num 24:24'], 
                 df['Deut 32:1' : 'Deut 32:43'], 
                 df['Deut 33:2' : 'Deut 33:29'], 
                 df['Jud 5:2' : 'Jud 5:31'], 
                 df['2 Sam 22:2' : '2 Sam 22:51'], 
                 df['Ps 18:1' : 'Ps 18:50'], 
                 df['Ps 68:2' : 'Ps 68:35']])

ABH['Stage'] = "ABH"

ABH['Text'] ['Exod 15:1'] = 'אָשִׁירָה לַיהוָה כִּי־גָאֹה גָּאָה סוּס וְרֹכְבוֺ רָמָה בַיָּם'
ABH['Text']['Num 23:7'] = 'מִן־אֲרָם יַנְחֵנִי בָלָק מֶלֶךְ־מוֺאָב מֵהַרְרֵי־קֶדֶם לְכָה אָרָה־לִּי יַעֲקֹב וּלְכָה זֹעֲמָה יִשְׂרָאֵל'
ABH['Text']['Num 23:18'] = 'קוּם בָּלָק וּשֲׁמָע הַאֲזִינָה עָדַי בְּנוֺ צִפֹּר'
ABH['Text']['Num 24:3'] = 'נְאֻם בִּלְעָם בְּנוֺ בְעֹר וּנְאֻם הַגֶּבֶר שְׁתֻם הָעָיִן'
ABH['Text']['Num 24:15'] = 'נְאֻם בִּלְעָם בְּנוֺ בְעֹר וּנְאֻם הַגֶּבֶר שְׁתֻם הָעָיִן'
ABH['Text']['Deut 33:2'] = 'יְהוָה מִסִּינַי בָּא וְזָרַח מִשֵּׂעִיר לָמוֺ הוֺפִיעַ מֵהַר פָּארָן וְאָתָה מֵרִבְבֹת קֹדֶשׁ מִימִינוֺ אֵ דָּת לָמוֺ'
ABH['Text']['2 Sam 22:2'] = 'יְהוָה סַלְעִי וּמְצֻדָתִי וּמְפַלְטִי־לִי'
ABH['Text']['Ps 18:2'] = 'אֶרְחָמְךָ יְהוָה חִזְקִי'

CBH = pd.concat([df['1 Sam 1:1' : '2 Sam 22:1'],
                 df['2 Sam 23:1' : '2 Kgs 25:30']])

CBH['Stage'] = "CBH"

TBH = pd.concat([df['Isa 40:1' : 'Isa 54:17'], 
                 df['Jer 1:1' : 'Ezek 48:35']])

TBH['Stage'] = "TBH"

LBH = pd.concat([df['Eccl 1:1' : 'Eccl 12:14'], 
                 df['Esth 1:1' : '1 Chr 9:44'], 
                 df['1 Chr 12:1' : '1 Chr 12:40'], 
                 df['1 Chr 15:1' : '1 Chr 15:24'], 
                 df['1 Chr 16:7' : '1 Chr 16:43'], 
                 df['1 Chr 21:1' : '1 Chr 29:19'], 
                 df['2 Chr 7:1' : '2 Chr 7:3'], 
                 df['2 Chr 14:9' : '2 Chr 15:7'], 
                 df['2 Chr 17:1' : '2 Chr 17:19'], 
                 df['2 Chr 21:12' : '2 Chr 21:17'], 
                 df['2 Chr 24:15' : '2 Chr 24:22'], 
                 df['2 Chr 26:6' : '2 Chr 26:21'], 
                 df['2 Chr 29:3' : '2 Chr 31:21'], 
                 df['2 Chr 33:10' : '2 Chr 33:20'], 
                 df['2 Chr 34:3' : '2 Chr 34:7'], 
                 df['2 Chr 36:22' : '2 Chr 36:23']])

LBH['Stage'] = "LBH"

max_leng = max(ABH.shape[0], CBH.shape[0], TBH.shape[0], LBH.shape[0])

ABH = pd.concat([ABH for i in range(math.ceil(max_leng/ABH.shape[0]))])[:max_leng] 
CBH = pd.concat([CBH for i in range(math.ceil(max_leng/CBH.shape[0]))])[:max_leng]
TBH = pd.concat([TBH for i in range(math.ceil(max_leng/TBH.shape[0]))])[:max_leng] 
LBH = pd.concat([LBH for i in range(math.ceil(max_leng/LBH.shape[0]))])[:max_leng]

COHeN_training_data = pd.concat([ABH, CBH, TBH, LBH])

COHeN_training_data.to_csv('COHeN_training_data.csv')

COHeN = load_dataset('csv', data_files='COHeN_training_data.csv')['train']
COHeN = COHeN.remove_columns('Verse')
COHeN = COHeN.train_test_split(test_size=.2, shuffle=True)
validation = COHeN['test'].train_test_split(test_size=.5, shuffle=True)
COHeN = DatasetDict({
    'train': COHeN['train'],
    'test': validation['test'],
    'eval': validation['train'],
})

stage = ClassLabel(num_classes=4, names=["ABH", "CBH", "TBH", "LBH"])
COHeN = COHeN.cast_column('Stage', stage)

COHeN.push_to_hub('gngpostalsrvc/COHeN')
