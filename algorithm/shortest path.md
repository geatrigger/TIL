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

  

