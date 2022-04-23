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
  
* tree 설치

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

  ```bash
  # bash, not sh
  printf "cpu temp\\t%.3f\\n" "$(</sys/class/thermal/thermal_zone0/temp)e-3"
  printf "gpu temp\\t%.1f\\n" "$(vcgencmd measure_temp | grep  -o -E '[[:digit:].]+')"
  ```

* 디스크 공간 확인

  ```bash
  df -h /
  tree -dh --du
  # big file (MB)
  du -am | sort -nr | head -20
  
  ```

* 인터넷 속도 테스트

  ```bash
  sudo apt-get install -y speedtest-cli
  speedtest --json
  ```

* 디스크 읽기 쓰기 속도 측정

* 와이파이 끄기

  ```shell
  sudo ifconfig wlan0 down
  ```

* 와이파이 속도 측정

  * https://forums.raspberrypi.com/viewtopic.php?t=280110
  * flirc 메탈 케이스는 와이파이 속도를 줄인다
  * cat6 선도 정상 허브도 정상
  * 허브에 연결을 느슨하게 함, 똑 소리나도록 밀어넣으니 100Mbit -> 1000Mbit로 증가
  * 공유기-허브 연결이 제대로 안되면 다른 ip대역 할당이됨
  * 허브에 랜선 연결은 똑소리나서 더이상 안들어갈 정도까지 넣음
  * iperf(2버전)으로 서로간의 인트라넷 속도 측정(3버전은 호환안됨)
  

