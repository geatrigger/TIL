# dijkstra algorithm

* 특정한 노드에서 출발하여 다른 모든 노드로 가는 최단 경로 계산

* 음의 간선이 없을 때 정상적으로 작동

* 그리디 알고리즘

  * 매 상황에서 가장 비용이 적은 노드를 선택

* 동작 과정

  * 출발 노드 설정
  * 최단 거리 테이블 초기화
  * 방문하지 않은 노드 중 최단 거리가 가장 짧은 노드 선택
  * 해당 노드(현재노드)를 거쳐 연결된 다른 노드로 가는 비용을 계산하여 최단 거리 테이블 갱신
  * 3, 4번 과정 반복

* 3번 과정을 할 때 for문으로 최단 거리가 가장 짧은 노드를 선택하면 시간복잡도가 O(V^2)이나 되지만, 해당 과정에서 우선순위큐를 사용하면 시간복잡도가  O(E log V)가 된다

  * E개의 원소를 우선순위 큐에 넣었다가 모두 빼내는 연산과 유사하여 O(E log E)
  * 중복 간선을 포함하지 않는 경우 시간복잡도는 O(E log E) -> O(E log V^2) ->  O(2E log V) -> O(E log V)

* 코드

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

  

  ```python
  import sys
  from heapq import heappush
  from heapq import heappop
  input = sys.stdin.readline
  
  max_value = 100000000
  
  def dijkstra(neighbors, start, V):
      distances = [max_value for x in range(V + 1)]
      node_h = [(0, start)]
      distances[start] = 0
      while node_h:
          d, node = heappop(node_h)
          # 이미 방문했던 노드는 스킵
          if distances[node] < d:
              continue
          for neighbor in neighbors[node]:
              neighbor_w, neighbor_node = neighbor
              cost = neighbor_w + d
              if cost > max_value:
                  cost = max_value
              if cost < distances[neighbor_node]:
                  distances[neighbor_node] = cost
                  heappush(node_h, (cost, neighbor_node))
      return distances
  
  
  V, E = map(int, input().split())
  K = int(input())
  
  neighbors = [[] for x in range(V + 1)]
  
  for i in range(E):
      u, v, w = map(int, input().split())
      neighbors[u].append((w, v))
  distances = dijkstra(neighbors, K, V)
  for i in range(1, V + 1):
      distance = distances[i]
      if distance >= max_value:
          print('INF')
      else:
          print(distance)
  ```



# Floyd-Warshall algorithm

* 모든 노드에서 다른 모든 노드까지의 최단 경로를 모두 계산

* 2차원 테이블에 최단 거리 정보 저장

* 다이나믹 프로그래밍 유형에 속함

* 각 단계마다 특정한 노드를 거쳐가는 게 빠른지, 기존에 구한 값이 빠른지 확인

  * a와b의 거리 = min(a와b의 거리, a와k의 거리 + k와b의 거리)

* N단계마다 O(N^2)의 연산을 통해 현재 노드를 거쳐가는 모든 경로를 고려한다

  * O(N^3)

* 코드

* https://www.acmicpc.net/problem/11404

  ```python
  # n : 도시의 개수
  # m : 버스의 개수
  # a, b, c : 시작도시, 도착도시, 비용
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

  

  ```python
  
  import sys
  input = sys.stdin.readline
  max_value = 100000000
  
  def floydwarshall(distance_m, n):
      for k in range(1, n+1):
          for i in range(1, n+1):
              for j in range(1, n+1):
                  distance_m[i][j] = min(distance_m[i][j], distance_m[i][k] + distance_m[k][j])
      return distance_m
  
  n = int(input())
  m = int(input())
  distance_m = [[max_value for x in range(n+1)] for y in range(n+1)]
  
  for i in range(1, n+1):
      distance_m[i][i] = 0
  for i in range(m):
      a, b, c = map(int, input().split())
      distance_m[a][b] = min(distance_m[a][b], c)
  
  distance_m = floydwarshall(distance_m, n)
  for i in range(1, n+1):
      for j in range(1, n+1):
          if distance_m[i][j] >= max_value:
              print(0, end=' ')
          else:
              print(distance_m[i][j], end=' ')
      print()
  ```

  

# Bellman-Ford algorithm

* 음수 간선의 순환이 있을 때 최단 경로를 구하는 법

* 기본 시간 복잡도는 O(VE)로 다익스트라보다 느리다

  * 다익스트라에선 각 과정에서 연결된 간선만 확인하는데 벨만 포드에선 각 과정에서 전체 간선을 확인하기 때문에 느리다
  * 벨만 포드 알고리즘은 다익스트라 알고리즘에서의 최적의 해를 항상 포함한다

* V-1번 과정에서 모든 간선의 수 E를 확인하고, 해당 간선에서 a, b, c(출발, 도착, 비용)의 정보가 있으면 a까지 가는 비용 + c이 b까지 가는 비용보다 작으면 갱신(단, a까지 가는 방법을 찾은상태여야 한다 = 비용이 inf가 아님)

* 마지막 N-1번째 시행(N번째 간선 확인)에서도 갱신이 일어난다면 음수 간선의 순환이 있다는 뜻이다

* 소스코드

  * https://www.acmicpc.net/problem/11657

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

    ```python
    import sys
    input = sys.stdin.readline
    max_value = 100000000
    
    def bellmanford(edges, N, start):
        distances = [max_value for x in range(N + 1)]
        distances[start] = 0
        for i in range(1, N+1):
            for edge in edges:
                A, B, C = edge
                new_distance = distances[A] + C
                if distances[A] != max_value and distances[B] > new_distance:
                    distances[B] = new_distance
                    if i == N:
                        return -1
    
        return distances
    
    N, M = map(int, input().split())
    edges = []
    for i in range(M):
        A, B, C = map(int, input().split())
        edges.append((A, B, C))
    distances = bellmanford(edges, N, 1)
    if distances == -1:
        print(-1)
    else:
        for i in range(2, N + 1):
            if distances[i] == max_value:
                print('-1')
            else:
                print(distances[i])
    ```

    