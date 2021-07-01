# tree 기본 용어

* root node : 부모가 없는 최상위 노드
* leaf node(단말 노드) : 자식이 없는 노드
* size : 트리에 포함된 모든 노드의 개수
* depth : 루트 노드부터의 거리
* height : depth 중 최댓값
* degree(차수) : 각 노드의 자식방향 간선 개수
* 트리의 크기가 N일 때 전체 간선의 개수는 N-1개

# 트리 순회(Tree Traversal)

* 전위 순회(pre-order traverse) : 루트 먼저 방문
* 중위 순회(in-order traverse) : 왼쪽 자식을 방문한 뒤 루트 방문
* 후위 순휘(post-order traverse) : 오른쪽 자식을 방문한 뒤에 루트 방문

```python
class Node:
    def __init__(self, data, left, right):
        self.data = data
        self.left = left
        self.right = right

def preorder(node):
    print(node.data, end=' ')
    if node.left:
        preorder(tree[node.left])
    if node.right:
        preorder(tree[node.right])

def inorder(node):
    if node.left:
        inorder(tree[node.left])
    print(node.data, end=' ')
    if node.right:
        inorder(tree[node.right])

def postorder(node):
    if node.left:
        postorder(tree[node.left])
    if node.right:
        postorder(tree[node.right])
    print(node.data, end=' ')


n = int(input())
tree = {}

for i in range(n):
    data, left, right = input().split()
    if left == 'None':
        left = None
    if right == 'None':
        right = None
    tree[data] = Node(data, left, right)

preorder(tree['A'])
print('')
inorder(tree['A'])
print('')
postorder(tree['A'])
print('')

'''
[예시 입력]
7
A B C
B D E
C F G
D None None
E None None
F None None
G None None
[예시 출력]
A B D E C F G 
D B E A F C G 
D E B F G C A 
'''
```

