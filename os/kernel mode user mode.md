# 커널모드와 유저모드가 따로 존재하는 이유

* https://jhnyang.tistory.com/190
* 악의적인 프로그램으로 의도치 않게 자료를 삭제하는 일을 막기 위해서(I/O 장치 보호)
* I/O 데이터에 관한 명령은 커널모드에서만 실행가능(유저모드는 안됨)

# 커널모드와 유저모드

* 커널모드(Kernel mode) : 운영체제(커널)가 수행되는 모드. privileged mode, supervisor mode, system mode, monitor mode라고도 불린다.
* 유저모드(User mode) : 어플리케이션 프로그램이 수행되는 모드
* mode bit로 커널모드, 유저모드를 구분하여 어셈블리 프로그램으로 강제로 out(I/O 디바이스 컨트롤러 레지스터에 값을 쓰는 명령어)을 실행해도 CPU가 mode bit를 보고 유저모드인 것을 파악하여 자기자신에게 exception을 걸어 해당 프로세스를 종료시킨다.
* system call
  * I/O 작업을 수행하려면 운영체제에 요청하는 방법인 system call을 사용해야 한다
  * user mode에서 read system call 발생 -> 
    trap to kernel mode(trap 인터럽트를 걸어 mode bit를 바꾸어 커널모드에 진입) -> 
    app state 저장 -> 
    trap handler에서 vector table을 찾아 read handler를 실행 ->
    app state 불러오기 ->
    user mode로 전환

