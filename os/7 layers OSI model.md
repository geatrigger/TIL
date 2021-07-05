# 7 layers OSI model

* https://www.forcepoint.com/ko/cyber-edu/osi-model
* https://ko.wikipedia.org/wiki/OSI_%EB%AA%A8%ED%98%95
* https://shlee0882.tistory.com/110
* Open Systems Interconnection Model
* 컴퓨터 네트워크 프로토콜 디자인과 통신을 계층으로 나누어 설명
* 각 계층은 하위 계층의 기능만을 사용하고, 상위 계층에게 기능을 제공한다
* Physical Layer
  * bit
  * 전기적으로 raw data를 주고받는 것에 대해 다룬다
  * voltage, pin layout, cabling, radio frequencies 등을 다룬다
* Data Link Layer
  * frame
  * node-to-node(Point to Point) 간 신뢰성있는 전송을 보장하기 위한 계층
  * mac address로 통신
  * 이더넷, HDLS, ADCCP, 패킷 스위칭 네트워크 프로토콜, LLC, ALOHA 사용
* Network Layer
  * packet
  * host-to-host(양 끝단, end-to-end)
  * ip address로 통신
  * 라우팅, 흐름 제어, segmentation, 오류제어, internetworking 수행
  * IP
* Transport Layer
  * segment
  * process-to-process
  * 패킷들의 전송이 유효한지 확인하고 전송 실패한 패킷들을 다시 전송
  * TCP, UDP
* Session Layer
  * 양 끝단의 응용프로세스가 통신을 관리하는 방법 제공
  * 세션 확립/유지/중단, 인증, 재연결
* Presentation Layer
  * 인코딩과 암호화작업
  * EBCDIC 인코딩 파일을 ASCII 인코딩 파일로 변환, 데이터가 text인지 gif인지 jpg인지 구분
* Application Layer
  * 응용프로그램, 응용서비스
  * 커뮤니케이션 파트너, 사용 가능한 자원량, 커뮤니케이션 동기화 등 명시
  * HTTP, FTP, SMTP, POP3, IMAP, Telnet