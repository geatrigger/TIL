# union find

* 서로소 집합(Disjoint Sets)
  
  * 공통 원소가 없는 두 집합
* 서로소 집합 자료구소
  * 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조
  * 합집합(Union) : 두 개의 집합을 하나의 집합으로 합치는 연산
  * 찾기(Find) : 특정 원소가 속한 집합을 알려주는 연산
* 효율
  * 합집합 연산이 편향되게 이루어지는 경우 찾기 함수 복잡도가 O(V)가 된다
  * 찾기 함수를 재귀적으로 호출한 뒤에 부모 테이블 값을 바로 갱신시키면 여러번 접근했을 때 더 적은 시간이 걸림

* 코드

  * https://www.acmicpc.net/problem/1647

  ```python
  # 해당 그래프에 사이클이 있는지 없는지 판단하기
  # 입력
  # v, e: 노드개수, 간선개수
  # a, b : 간선을 이루는 두 노드 a, b
  3 3
  1 2
  2 3
  1 3
  # 출력
  사이클발생
  ```

  ```python
  def find_root(parents, node):
      parent = node
      while parent != parents[node]:
          parent = parents[node]
      parents[node] = parent
      return parent
  
  def union(parents, a_root, b_root):
      if a_root < b_root:
          parents[b_root] = a_root
      else:
          parents[a_root] = b_root
  
  v, e = map(int, input().split())
  parents = [i for i in range(v + 1)]
  cycle = False
  for i in range(e):
      a, b = map(int, input().split())
      a_root = find_root(parents, a)
      b_root = find_root(parents, b)
      if a_root == b_root:
          cycle = True
      else:
          union(parents, a_root, b_root)
  if cycle:
      print('사이클 존재')
  else:
      print('사이클 없음')
  ```
  
  