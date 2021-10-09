# hdfs structure

* NameNode
  * https://hadoop.apache.org/docs/r3.3.1/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html#NameNode_and_DataNodes
  * HDFS의 namespace 관리
    * 관습적으로 사용되는 계층 구조로 되어 있다
    * user quotas, access permissions 지원
    * hard link, soft link는 지원 안함(누군가가 구현하면 차용하겠지만, 현재 지원안함)
    * 데이터의 무결성을 보장하기 위해 namespace에 각 block에 대한 checksum을 저장한다.
  * 클라이언트의 파일 접근 관리
  * replication factor가 보관되어 있다
  * Replication Placement
    * Hadoop Rack Awareness를 이용해 DataNode가 어느 rack id에 속한지 결정한다
    * replica들을 모두 다른 rack에 두면 rack간의 이동비용이 크기 때문에 write의 비용이 증가해서 선호하지 않는다.
    * 안정성과 성능을 위해, replication factor가 3일때, 각각의 replica들은 local machine(write가 있는 곳), 같은 rack에 속한 다른 임의의 DataNode, 다른 rack에 속한 DataNode에 저장된다(data reliability와 read 성능을 줄이지 않음)
    * replication factor가 3이 넘어가면 그 이후 replica의 저장 위치는 랜덤이다(단, rack당 replica개수가 (replicas - 1) / racks + 2를 넘지 말아야 한다)
    * 최대 replica개수는 DataNode의 개수와 같다(같은 DataNode에 replica를 두지 않기 때문)
  * Replica Selection
    * global bandwidth 소모와 read latency를 줄이기 위해서, HDFS는 reader입장에서 가장 가까운 replica 하나만 읽는다
  * Re-Replication
    * DataNode가 사용불가능해지거나, replica가 망가지거나, DataNode에 있는 hard dick가 망가지거나, replication factor가 늘어나면 수행한다.
    * DataNode가 죽었다고 판단하는 것은 신호가 안오기 시작한지 기본적으로 replication storm을 피하기 위해 10분 뒤에 한다. 이 간격을 짧게 설정할 수 있다
  * FsImage, EditLog등의 메타데이터가 망가지면 안되기 때문에 해당 메타데이터들을 복제본을 만들어서 저장한다. FsImage, EditLog는 동기적으로 업데이트 된다. NameNode가 재시작하면 가장 최근에 일치하는 FsImage, EditLog가 사용된다.
* DataNode

  * NameNode가 지정해준 블록 저장, 삭제, 복제
  * 주기적으로 Heartbeat(정상작동중이라는 신호)와 Blockreport(해당 DataNode의 모든 블록정보)를 NameNode에 보낸다
* EditLog
  * 마지막 checkpoint 이후로 namespace(file system)에 변화 log
* FsImage
  * 파일이름, 경로, block 개수, slave 정보 등 메타데이터들을 image 파일 형태로 저장한 것
  * namespace의 가장 마지막 checkpoint
* Secondary NameNode
  * https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsUserGuide.html#Secondary_NameNode
  * NameNode가 시작할 때 fsimage를 통해 HDFS 상태를 읽어오고 edit log들과 합치는데, edit log들이 너무 많이 쌓이면 처음 실행이 매우 느려진다.
  * 따라서 fsimage와 edit log를 주기적으로 합치고 edit log 크기를 limit을 넘지 않도록 조절한다
  * NameNode와 다른 기기에서 실행한다
  * 가장 최근 checkpoint image를 저장하고 primary NameNode가 필요할 때 항상 준비되어 있다
  * Checkpoint Node와 비교
    * https://data-flair.training/forums/topic/differentiate-secondary-namenode-and-checkpoint-node-in-hadoop/
    * 똑같이 fsimage와 edit log들을 합치는 일을 수행한다
    * 다만 Checkpoint Node의 경우 merge한 다음 NameNode에게 보내지만, Secondary NameNode의 경우 영구 저장소에 저장이 되었다가 Namenode가 고장났을 때 저장소에서 fsimage 형태로 가져올 수 있다

