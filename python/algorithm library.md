# algorithm library

* 내장 함수

  * sum, min, max
  * sorted
    * `sorted(array, key=lambda x: x[0], reverse=True)`

* itertools

  * permutations

    * 완전탐색 유형 문제 코드 간결하게 만들어줌

    * 서로 다른 n개에서 서로 다른 r개를 선택하여 일렬로 나열

    * nPr

      ```python
      from itertools import permutations
      ```

      

  * combinations

    * 완전탐색 유형 문제 코드 간결하게 만들어줌
    * 서로 다른 n개에서 순서에 상관없이 서로 다른 r개를 선택하는 것
    * nCr

* heapq

* bisect

  * 이진 탐색

* collections

  * deque

  * Counter

    * iterable 객체에서 각 원소의 등장 횟수를 세는 기능

      ```python
      from collections import Counter
      ```

* math

  * gcd
  * factorial
  * sqrt