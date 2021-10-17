# hadoop version

* https://wikidocs.net/26170
* 버전1
  * 하둡의 기본 아키텍처 정립
  * MapReduce는 JobTracker, TaskTracker가 담당
  * HDFS는 NameNode와 DataNode가 담당
  * JobTracker가 클러스터의 자원관리, 애플리케이션의 라이프사이클 관리를 모두 담당하여 병목현상 발생
* 버전2
  * YARN 아키텍처 도입으로 잡트래커의 기능 분리
    * Resource Manager와 Node Manager가 클러스터의 자원관리
    * Application Master와 Container가 애플리케이션 라이프 사이클 관리
* 버전3
  * https://hadoop.apache.org/docs/r3.3.1/
  * Erasure Coding
    * 패리티 블록을 사용하여  기존에 1G파일을 3G파일로 복제해서 저장할 때 갖는 안정성을 유지하면서 디스크를 덜 사용하게 되어 1G파일을 1.5G만 사용하여 저장이 가능해졌다
  * YARN 타임라인 서비스 v2
  * 하둡 v1부터 사용하던 쉘스크립트 다시 작성하여 버그 해결
  * 네이티브 코드를 수정하여 셔플단계의 처리 속도 증가
  * 높은 장애허용성을 위해 2개 이상의 네임노드 지원
  * 기본포트 중 Linux ephemeral port 범위에 있어서 다른 실행 프로그램과 종종 충돌을 일으켰던 것을 그렇지 않은 범위로 바꿈
  * Intra-datanode balancer를 추가하여 DataNode안에서의 skew현상을 방지함(Inter datanode끼리의 데이터쏠림현상은 기존의 HDFS balancer가 해결)