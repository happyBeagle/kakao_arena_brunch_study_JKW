# User2Vec

# Code

```
$> tree -d
.
├── dataprocessing.py
├── user2vec.py
└── inference.py
```

# Recommended Strategies
User2Vec을 사용한 콘텐츠 기반 필터링의 알고리즘은 다음과 같습니다.
1. 브런치에 작성된 문서들을 바탕으로 Doc2Vec 학습
2. 독자라면 조회한 문서들의 Vector의 평균으로, 작가라면 작성한 문서들의 Vector의 평균으로 User2Vec 학습
3. 독자가 구독한 작가가 중첩 기간에 작성한 글을 최신 순으로 추천
4. 중첩 기간 동안 독자가 조회한 작가가 중첩 기간에 작성한 글을 최신 순으로 추천
5. User2Vec으로 유사도가 높은 작가가 중첩 기간에 작성한 글을 최신 순으로 추천
6. 중첩 기간 동안 조회한 글은 제외
7. 3~5 순으로 100개의 글이 추천 되면 종료
8. 만약 Cold Start인 경우, 중첩 기간 동안의 조회 순으로 상위 100개의 글을 추천

# Result

<img width="1025" alt="스크린샷 2021-04-13 20 15 05" src="https://user-images.githubusercontent.com/55614265/114550812-9778b180-9c9d-11eb-8c8b-dffeca802d3e.png">
