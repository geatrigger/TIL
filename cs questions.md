# 자료구조
# 알고리즘

* 트리 순회(tree traversal) 종류 세가지 알고리즘에 대해서 코드로 구현하고 다음 트리가 예시로 주어질 때 각각 어떤 결과가 나오는가

  ```python
  # data, left, right를 입력받는다
  7
  A B C
  B D E
  C F G
  D None None
  E None None
  F None None
  G None None
  # 출력
  A B D E C F G 
  D B E A F C G 
  D E B F G C A 
  ```

* fenwick알고리즘 구현

  ```python
  # 입력
  # N, M, K : 수의 개수, 수의 변경이 일어나는 횟수, 구간의 합을 구하는 횟수
  # N개의 수
  # a, b, c : a가 1일 경우 b번째 수를 c로 바꿈, a가 2일 경우 b~c번째 구간의 합 출력
  5 2 2
  1
  2
  3
  4
  5
  1 3 6
  2 2 5
  1 5 2
  2 3 5
  # 출력
  17
  12
  ```

  

* 표현하는 수의 범위가 비교적 작을 때 쓰는 O(N)에 가까운 복잡도를 가진 알고리즘 구현

  ```python
  # 입력
  [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
  # 출력
  [0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9]
  ```

  

* quick sort 구현

  * quick sort의 시간복잡도
  * quick sort의 시간복잡도가 최악인 경우는 언제 나오는가

  ```python
  # 입력
  1 3 4 2 5, 2 3 1 4 5
  # 출력
  1 2 3 4 5, 1 2 3 4 5
  ```

  -------------------------------------
  
* dfs, bfs 구현

  * 뜻

  ```python
  # 입력
  graph = [
    [],
    [2, 3, 8],
    [1, 7],
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7]
  ]
  # 출력 dfs
  1 2 7 6 8 3 4 5
  # 출력 bfs
  1 2 3 8 7 4 5 6 
  ```

* shortest path

  * dijkstra algorithm이 어떻게 쓰이는지 말하고 구현

    * 시간복잡도
  
    ```python
    # 방향그래프
    # 정점의 개수 V, 간선의 개수 E
    # 시작정점 K
    # u v w (u에서 v로 갈 때 가중치 w인 간선 존재)
    # 입력
    5 6
    1
    5 1 1
    1 2 2
    1 3 3
    2 3 4
    2 4 5
    3 4 6
    # i번째 줄에 i번째까지 가는데 드는 비용 출력
    # 출력
    0
    2
    3
    7
    INF
    ```

    

  * floyd-warshall algorithm이 어떻게 쓰이는지 말하고 구현
  
    * 시간복잡도
    
    ```python
    # n : 도시의 개수
    # m : 버스의 개수
    # a, b, c : 시작도시, 도착도시, 비용
    # 갈 수 없는 곳은 0으로 출력
    # 도시에서 도시로 가는 경로는 여러 개 있을 수 있다.
    # 임력
    5
    14
    1 2 2
    1 3 3
    1 4 1
    1 5 10
    2 4 2
    3 4 1
    3 5 1
    4 5 3
    3 5 10
    3 1 8
    1 4 2
    5 1 7
    3 4 2
    5 2 4
    # 출력
  0 2 3 1 4
    12 0 15 2 5
  8 5 0 1 1
    10 7 13 0 3
    7 4 10 6 0
    ```
    
  * bellman-ford algorithm이 어떻게 쓰이는지 말하고 구현
  
    * 시간복잡도
  
    ```python
    # 입력
    # N, M : 도시의 개수, 버스 노선의 개수
    # A, B, C : 시작도시, 도착도시, 버스를 타고 이동하는데 걸리는 시간
    # 1번 도시에서 다른 도시에 가는데 걸리는 시간 
    # 어떤 도시에 가는데 시간을 무한히 오래 전으로 되돌릴 수 있으면 -1만 출력
    # 그 외엔 걸리는시간을 도시 순서대로 출력, 해당 도시로 가는 경로가 없으면 -1 출력
    # 1
    3 4
    1 2 4
    1 3 3
    2 3 -1
    3 1 -2
    #
    4
    3
    # 2
    3 4
    1 2 4
    1 3 3
    2 3 -4
    3 1 -2
    #
    -1
    # 3
    3 2
    1 2 4
    1 2 3
    #
    3
    -1
    ```
  

----------------------------

* 주어진 그래프에서 사이클이 있는지 확인하는 코드짜기

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

  

# OS

* LRU알고리즘
  * cache에서 쓰이는 이유
  * 구현
    * cache크기 4
    * 참조 순서 : 1 2 3 1 4 5
    * 실행 후 cache : 5 4 1 3

------------------------

* 7 layers OSI model
  * layer모델의 특징
  * 7계층 각각의 특징
* Kernel mode and User mode
  * 커널모드와 유저모드로 나누는 이유
  * system call의 작동과정 설명
* Hypervisor and Container
  * 가상화의 특징
  * Hypervisor와 Container의 차이점
* Microkernel and Monolithickernel
  * kernel의 역할
  * Microkernel 와 Monolithickernel에서 커널, 주요 서비스의 실행방법
  * Microkernel 와 Monolithickernel 장단점
* Interrupt 와 Trap의 차이점

--------------------------------------



# DB
# 컴퓨터구조
# 네트워크
# 확률 통계