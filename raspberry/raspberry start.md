# raspberry essential

* raspbian 64bit(aarch64==armv8, devian계열)

* 키보드 입력 설정
  * https://dullwolf.tistory.com/17
  * raspberry 400 us자판에서 올바른 키세팅을 사용하려면 Korean말고 US선택
  
* 와이파이 연결
  * 라즈베리식 wifi 설정
    * https://webnautes.tistory.com/903
  * linux iwlist wlan0 scan, /etc/wpa_supplicant/wpa_supplicant.conf
    * https://webnautes.tistory.com/141
  
* raspberry package 설치

  * https://m.blog.naver.com/itperson/220653088291

  ```shell
  # installed list
  apt --installed list
  # install vim
  apt-get install -y vim
  # package information update
  apt-get update
  # can be upgraded list
  apt-get --just-print upgrade
  # search package(start with vim)
  apt-cache search ^vim
  # delete package with config(delete vim)
  apt-get --purge remove vim
  # add repository
  apt-add-repository ppa:<repository address>
  ```

* vim 설치

* sudoer 추가

  * /etc/sudoers에 추가

    ```shell
    pi	ALL=(ALL) NOPASSWD:ALL
    zeppy ALL=(ALL) NOPASSWD:/bin/mkdir,/bin/rmdir
    ```

* password 설정

* ssh 설정

  * [ssh](../ssh/ssh.md)

# raspberry virtualization

* 라즈베리용 도커 설치

  * [raspberry docker](./raspberry docker.md)
* 도커 sudo 없이 실행설정
* 도커 멀티 플랫폼 빌드(buildx)
* 쿠버네티스 설치(k3s)

# raspberry development environment

* DDNS 설정으로 외부에서의 접근

  * 중간에 모뎀을 거친다면
    * https://itfix.tistory.com/163
  * vpn 접속 후 로컬ip로 접근
* git 설치
* java 설치
* 호스트 설정
  * ip고정

# raspberry status check

* 온도 모니터링
