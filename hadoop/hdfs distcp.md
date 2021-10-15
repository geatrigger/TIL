# hdfs distcp

* https://hadoop.apache.org/docs/stable/hadoop-distcp/DistCp.html

* https://wikidocs.net/25261

* 용도
  * distributed copy
  * inter/intra-cluster copying에 쓰이는 도구
  
* hadoop fs -cp와의 차이
  
  * cp 명령어는 파일을 하나씩 복사하지만, DistCp는 맵리듀스를 이용하여 대규모의 파일을 병렬로 복사한다
  
* 사용법

  * -update : 복사시 파일 이름, 사이즈를 비교해서 복사

  * -overwrite : 기존 복사 파일을 삭제하고 덮어씀

  * -f : 복사 source 위치가 여러개일 때 파일로 전달

  * -m : 매퍼 개수 설정

  * -D : 하둡 옵션 전달

    ```shell
    # a 폴더를 b 로 복사 
    $ hadoop distcp hdfs:///user/a hdfs:///user/b
    
    # a, b 폴더를 c로 복사 
    $ hadoop distcp hdfs:///user/a hdfs:///user/b hdfs:///user/c
    
    # -D 옵션을 이용하여 큐이름 설정 
    $ hadoop distcp -Dmapred.job.queue.name=queue hdfs:///user/a hdfs:///user/b
    
    # -D 옵션으로 메모리 사이즈 전달 
    $ hadoop distcp -Dmapreduce.map.memory.mb=2048 hdfs:///user/a hdfs:///user/b
    
    # 파일이름, 사이즈를 비교하여 변경 내역있는 파일만 이동 
    $ hadoop distcp -update hdfs:///user/a hdfs:///user/b hdfs:///user/c
    
    # 목적지의 파일을 덮어씀 
    $ hadoop distcp -overwrite hdfs:///user/a hdfs:///user/b hdfs:///user/c
    ```

    

