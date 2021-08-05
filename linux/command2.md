# file



# ls -F



#  mkdir -p m1/m2



# rm -rf color



# more

* j

* q

  ```shell
  more /etc/services
  ```

  

# less

* j, k
* ctrl+f
* ctrl+b : 뒤로

# tail

# head

# cat /etc/hosts

* 네트워크?

# cp -i

* 복사 붙여넣기 할 때 기존의 파일 덮어씌울 건지 묻기

# ln

* ln a alink
* ln -s a alink
* hard link의 경우 inode값이 같게 나온다

# ls -il

* inode 출력

# date

* 현재 시간 출력

# grep

* 정규식으로 파일 내용 검색
* grep ff /etc/hosts

# find

* 조건에 맞는 파일 찾기

# which

* 명령어, 프로그램을 찾아주는 명령어

# /etc/services

* 'NETBIOS'가 있는 행을 찾아 행번호와 함께 출력
  * grep -n NETBIOS /etc/services

# vim

* :set number
  * 몇 번째 줄인지 보여줌
  * 그 후 숫자 누른 다음 G를 누르면 해당 번째 줄로 이동

# yy

* 행 복사

# p, P

* 붙여넣기

# ?like, :/like

* 이상태에서 n을 누르게되면 이전/다음 like 단어에 커서 이동
* :s/like/Like를 하면 해당 행에서 like 단어를 Like로 바꿔줌
* :.,-1s/like/Like를 하면 현재커서 기준 ~만큼 바꿈

# tar

* tape archive
* 파일을 묶어서 하나로 만든 것
* 묶기 : tar cvf unix.tar Unix
* 풀기 : tar xvf unix.tar
* 아카이브 내용 확인 : tar tvf myfold.tar
* 아카이브 내용 업데이트 : tar uvf ch2.tar ch2
  * 이미 생성한 것을 수정하는 역할
* 아카이브 내용 추가 : tar rvf ch2.tar hosts
  * 기존 아카이브에 파일 추가
* 묶고 압축 : tar cvzf ch2.tar.gz ch2
* 압축 풀기 : tar xvzf ch2.tar.gz

