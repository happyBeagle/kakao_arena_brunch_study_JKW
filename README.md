# kakao_arena_brunch_study_JKW

```
$> tree -d
.
├── data
│     ├── /contents
│     ├── /predict
│     ├── /read
│     ├── magazine.json
│     ├── metadata.json
│     └── users.json
├── /data_processing
│     ├── EDA_data.ipynb
│     ├── EDA_magazine.ipynb
│     ├── EDA_metadata.ipynb
│     ├── EDA_predict.ipynb
│     ├── EDA_read.ipynb
│     └── EDA_users.ipynb
├── /preprocessing
│     ├── readrawdata.py
│     └── singleton.py
├── /model
│     ├── /user2vec
│     │    ├── dataprocessing.py
│     │    ├── user2vec.py
│     │    └── inference.py
│     ├── /ConvMF
│     │    ├── dataprocessing.py
│     │    ├── convmf.py
│     │    ├── convmodel.py
│     │    └── tokenizer.py
│     ├── /mf
│     │     ├── mf.py
│     │     └── inference.py
│     └── /sasrec
│          ├── /data_loader
|          │    └── dataset.py
│          ├── /model
|          │    ├── loss.py
|          │    └── model.py
│          ├── /trainer
|          │    └── trainer.py
│          ├── config.py
│          ├── datapreprocessing.py
│          ├── inference.py
│          ├── train.py
│          └── utils.py
├── /recommend
│     └── recommend.txt : 제출 파일
└── util.py : 그 외 필요한 기능
``` 
