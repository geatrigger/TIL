# kernel의 역할

* https://www.guru99.com/microkernel-in-operating-systems.html
* https://en.wikipedia.org/wiki/Microkernel
* 시스템 자원을 관리함(소프트웨어와 하드웨어 사이의 다리역할)
* bootloader다음에 실행되는 첫 프로그램

# Microkernel

* os를 실행시키기위한 최소한의 소프트웨어(memory management, processor scheduling, inter-process communication)만 priviledged level에서 실행
* 그 외의 다른 중요한 기능들은 따로 분리되어 user mode에서 실행됨(file system servers, device driver servers, networking servers, [display servers](https://en.wikipedia.org/wiki/Display_server), and user interface device servers)
* 상대적으로 많은 코드가 필요하다
* 서비스에 이상이 생겨도, microkernel에는 영향을 주지 않는다
* server끼리는 IPC로 통신한다
* 상대적으로 성능이 떨어진다
* L4Linux, QNX, SymbianK42, Mac OS X, Integrity

# Monolithic Kernel

* 모든 기본 시스템 서비스가 하나의 프로그램으로 실행이 된다
* 상대적으로 적은 코드가 필요하다
* 서비스에 이상이 생기면 monolithic kernel전체에 오류가 생긴다
* 하나의 binary file이라 통신이 더 편하다
* 상대적으로 성능이 좋다
* Linux, BSDs, Microsoft Windows (95,98, Me), Solaris, OS-9, AIX, DOS, XTS-400