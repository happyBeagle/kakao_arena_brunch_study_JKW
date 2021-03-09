# kakao_arena_brunch_study_JKW

```
$> tree -d
.
├── config.py : 경로 설정
├── preprocessing.py : model 학습을 위한 전처리 (통합), 주석 처리
├── /model
│      ├── model_1.py : 김남혁 님의 추천 시스템
│      ├── model_2.py : 우종빈 님의 추천 시스템
│      └── model_3.py : 조민정 님의 추천 시스템
├── train.py
├── ensemble.py : model_1.py + model_2.py + model_3.py
├── evaluate.py : MAP, NDCG, EntDiv
├── main.py
├── /recommend
│      └── requirements.txt : 제출 파일
└── util.py : 그 외 필요한 기능
``` 
