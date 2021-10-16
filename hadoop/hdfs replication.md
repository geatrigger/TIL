# hdfs replication

* https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html
* replication 배치방법(replication factor가 3인경우)
  * 안정성을 위해 모두 다른 rack에 replica를 두는 것은 여러 rack에 block들을 전달하는 비용이 커서 좋지 않음
  * 첫째 replica는 writer가 datanode 위에 있는 경우, 해당 datanode에 올림 / 그렇지 않다면 writer와 같은 랙에 있는 랜덤 노드에 올림
  * 둘째 replica는 writer와 다른 rack의 노드에 올림
  * 셋째 replica는 둘째 replica와 같은 rack의 다른 노드에 올림
  * 이러한 전략은 모두 다른 rack에 두는것보다 rack사이에서 write하는 traffic을 줄여준다
  * rack failure가 node failure보다 확률이 적어서 reliability에 영향 없음
  * replication factor가 3보다 큰 경우, 4번째 replica부터는 랜덤한 곳에 저장된다(단, 하나의 rack에는 최대 (replicas - 1) / racks + 2개의 replica가 존재할 수 있다)
    * 4가지의 HDFS 블록 배치 전략이 존재한다
    * https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsBlockPlacementPolicies.html
  * HDFS는 global bandwidth 소모를 줄이고 지연시간을 줄이기 위해 최대한 reader와 가까운 replica를 읽으려고 한다(same rack > different rack, local data center > remote data center)

