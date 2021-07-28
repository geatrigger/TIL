# max data size

* 라즈베리파이에서 mongo 32bit의 경우 최대 데이터 저장 용량이 2GB라는 것을 듣고, 32bit mysql도 그런지 확인해보았다

* mysql에서 입력

  ```mysql
  show table status from testdb;
  +------------+--------+---------+------------+------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-------------------+----------+----------------+---------+
  | Name       | Engine | Version | Row_format | Rows | Avg_row_length | Data_length | Max_data_length | Index_length | Data_free | Auto_increment | Create_time         | Update_time | Check_time | Collation         | Checksum | Create_options | Comment |
  +------------+--------+---------+------------+------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-------------------+----------+----------------+---------+
  | test_table | InnoDB |      10 | Compact    |    0 |              0 |       16384 |               0 |            0 |  10485760 |           NULL | 2021-07-28 04:38:03 | NULL        | NULL       | latin1_swedish_ci |     NULL |                |         |
  +------------+--------+---------+------------+------+----------------+-------------+-----------------+--------------+-----------+----------------+---------------------+-------------+------------+-------------------+----------+----------------+---------+
  show global status like 'innodb_page%';
  +----------------------+-------+
  | Variable_name        | Value |
  +----------------------+-------+
  | Innodb_page_size     | 16384 |
  | Innodb_pages_created | 5     |
  | Innodb_pages_read    | 144   |
  | Innodb_pages_written | 28    |
  +----------------------+-------+
  4 rows in set (0.00 sec)
  ```

* 확인 결과 엔진의 종류가 MyISAM, InnoDB가 있는데 MyISAM의 경우 OS에 따라서 테이블의 최대 데이터 저장용량이 달라진다

  * MyISAM의 경우

    ```
    Operating System                      File-size Limit
    -------------------------------------------------------------
    Win32 w/ FAT/FAT32                    2GB/4GB
    Win32 w/ NTFS                         2TB (possibly larger)
    Linux 2.2-Intel 32-bit                2GB (LFS: 4GB)
    Linux 2.4+  (using ext3 file system)  4TB
    Solaris 9/10                          16TB
    OS X w/ HFS+                          2TB
    ```

  * InnoDB의 경우

    | InnoDB Page Size | Maximum Tablespace Size |
    | :--------------- | :---------------------- |
    | 4KB              | 16TB                    |
    | 8KB              | 32TB                    |
    | 16KB             | 64TB                    |
    | 32KB             | 128TB                   |
    | 64KB             | 256TB                   |

* 라즈베리파이에 깔린 MySQL의 경우, page size가 16KB이므로 테이블 최대 저장용량은 64TB이다