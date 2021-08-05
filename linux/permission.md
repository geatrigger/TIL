# 권한

* 소유자, 소유자그룹, 기타 순으로 권한 쓰여짐
  * rw-r--r--
* chmod
  * 접근 권한을 변경하는 방법
  * chmod u-w test.txt
    * 소유자의 쓰기 권한 제거
  * chmod u=rwx test.txt
    * 전에 어땠던 간에 해당권한으로 설정
    * 읽기, 쓰기, 실행 가능
  * chmod u-x+w test.txt
    * execute 제거, write 추가
  * 숫자로도 표현가능
    * 4: r--
    * 2: -w-
    * 1: --x
  * chmod u=r,g=rwx,o=rx test.txt
    * 475권한 부여
    * chmod 475 test.txt

