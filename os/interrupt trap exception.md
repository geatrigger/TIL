http://melonicedlatte.com/computerarchitecture/2019/02/12/213856.html

https://pediaa.com/difference-between-trap-and-interrupt/

# interrupt

* Asynchronous Interrupt : 비동기식 인터럽트, 하드웨어 인터럽트로 정해진 기준 없이 예측 불가하게 발생하는 이벤트
* 인터럽트가 발생하면 프로세서는 현재 실행중인 process의 state를 저장하고 interrupt handler로 전환한다. interrupt처리가 끝난 후 원래의 state를 불러온다
* I/O interrupt, keyboard event, network packet arrived, timer ticks


# trap

* Synchronous Interrupt : 동기식 인터럽트, 기준에 맞추어 os의 기능을 수행시키는 이벤트
* 실행 중인 프로그램 내에 테스트를 위해 특별한 조건을 걸어 놓아 system call이 발생되면 trap이 catch를 함
* 그리고 trap handler가 각 상황에 맞게 각 서비스 루틴 또는 핸들러에서 처리를 시킨다

