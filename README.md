# kakao_arena_brunch_study_JKW

```
$> tree -d
.
├── config.py : 경로 설정
├── data
│      ├── /contents
│      ├── /predict
│      ├── /read
│      ├── magazine.json
│      ├── metadata.json
│      ├── users.json
├── preprocessing.py : model 학습을 위한 전처리 (통합), 주석 처리
├── /model
│      ├── /user2vec
│      │      ├── user2vec.py
│      │      └── inference.py
│      ├── /model
│      │      ├── model.py 
│      │      └── inference.py
│      └── /mf
│             ├── mf.py
│             └── inference.py
├── ensemble.py : model_1.py + model_2.py + model_3.py
├── evaluate.py : MAP, NDCG, EntDiv
├── main.py
├── /recommend
│      └── requirements.txt : 제출 파일
└── util.py : 그 외 필요한 기능
``` 
