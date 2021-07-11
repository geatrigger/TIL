# dfs

* 깊이 우선 탐색

* 탐색 시작 노드를 스택에 삽입하고 방문 처리

* 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문처리

* 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 제거

* 스택이 비면 종료

* 재귀를 이용하거나, 스택을 이용하거나

* 소스코드

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
  # 출력
  1 2 7 6 8 3 4 5
  ```

  

  ```python
  def dfs(graph, start):
      stack = [start]
      visited = [0 for x in graph]
      while stack:
          node = stack.pop()
          if visited[node]:
              continue
          print(node, end=' ')
          visited[node] = 1
          for child in graph[node][::-1]: # 가장 작은 값이 스택의 마지막에 오도록
              if visited[child]:
                  continue
              stack.append(child) # 여기서 visited[child] = 1을 하면 dfs가 아니게 됨
              
  # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
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
  
  dfs(graph, 1)
  ```



# bfs

* 너비 우선 탐색

* 탐색 시작 노드를 큐에 삽입하고 방문 처리

* 큐에서 노드를 꺼낸 뒤 해당 노드의 인접 노드 중 방문하지 않은 노드를 모두 큐에 삽입하고 방문처리

* 소스코드

  ```python
  # 출력
  1 2 3 8 7 4 5 6 
  ```

  

  ```python
  from collections import deque
  
  def bfs(graph, start):
      queue = deque([start])
      visited = [0 for x in range(len(graph) + 1)]
      while queue:
          node = queue.popleft()
          if visited[node]:
              continue
          visited[node] = 1
          print(node, end=' ')
          for child in graph[node]:
              if visited[child]:
                  continue
              queue.append(child)
  
  # 각 노드가 연결된 정보를 리스트 자료형으로 표현(2차원 리스트)
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
  
  bfs(graph, 1)
```
  
  