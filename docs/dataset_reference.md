# Cybersecurity Dataset Specifications and Feature Reference

## 1. Primary Benchmark: UNSW-NB15

The machine learning pipeline evaluates network traffic using the authentic UNSW-NB15 benchmark dataset, created by the Australian Centre for Cyber Security (ACCS) at the University of New South Wales Canberra.

- Purpose: Created to overcome limitations of older intrusion datasets (such as KDD98 and NSL-KDD) by capturing modern network protocol behaviors, realistic benign traffic, and low-footprint attack activities generated via the IXIA PerfectStorm tool.
- Official Partitioning:
  - Official Training Set (`data/raw/UNSW_NB15_training-set.csv`): 175,341 records (119,341 attack flows, 56,000 normal flows).
  - Official Testing Set (`data/raw/UNSW_NB15_testing-set.csv`): 82,332 records (45,332 attack flows, 37,000 normal flows).
- Target Representations:
  - Binary Label (`label`): `0` = Benign / Normal traffic; `1` = Malicious / Attack traffic.
  - Threat Category (`attack_cat`): Attack families including `Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, and `Worms`. Excluded from model features to prevent label leakage.

---

## 2. Authentic Benchmark vs. Demonstration Sample

| Dimension | Authentic UNSW-NB15 Benchmark | Demonstration Sample (`data/sample/unsw_sample.csv`) |
| :--- | :--- | :--- |
| File Location | `data/raw/UNSW_NB15_training-set.csv` and `testing-set.csv` | `data/sample/unsw_sample.csv` |
| Record Volume | 175,341 train records / 82,332 test records | 100 sample records |
| Evaluation Purpose | Statistically rigorous benchmark evaluation. | Fast smoke testing and automated CI verification. |
| Version Control | Excluded from Git via `.gitignore`. | Tracked in Git for instant test execution. |
| Execution Command | `python src/train.py` | `python src/train.py --sample` |

---

## 3. UNSW-NB15 Comprehensive Feature Dictionary

The UNSW-NB15 dataset contains 49 features grouped into five logical categories:

### 3.1 Flow Identifiers (Excluded from Model Features to Prevent Memorization)
- `id`: Sequence identifier integer (strictly dropped).
- `srcip`: Source IP address (dropped if present in raw captures).
- `sport`: Source port number (dropped if present).
- `dstip`: Destination IP address (dropped if present).
- `dsport`: Destination port number (dropped if present).

### 3.2 Basic Flow Features
- `dur` (*float*): Record total duration in seconds.
- `proto` (*nominal*): Transaction protocol (e.g., `tcp`, `udp`, `arp`, `icmp`).
- `service` (*nominal*): Application service (e.g., `http`, `ftp`, `smtp`, `ssh`, `dns`, `-` mapped to `unknown`).
- `state` (*nominal*): Connection state indicator (e.g., `FIN`, `INT`, `CON`, `REQ`, `RST`).
- `spkts` (*integer*): Source to destination packet count.
- `dpkts` (*integer*): Destination to source packet count.
- `sbytes` (*integer*): Source to destination transaction bytes.
- `dbytes` (*integer*): Destination to source transaction bytes.
- `rate` (*float*): Overall packet transmission rate (packets/sec).

### 3.3 Content Features
- `sttl` (*integer*): Source to destination time-to-live value.
- `dttl` (*integer*): Destination to source time-to-live value.
- `sload` (*float*): Source bits per second.
- `dload` (*float*): Destination bits per second.
- `sloss` (*integer*): Source packets retransmitted or dropped.
- `dloss` (*integer*): Destination packets retransmitted or dropped.
- `sinpkt` (*float*): Source inter-packet arrival time (milliseconds).
- `dinpkt` (*float*): Destination inter-packet arrival time (milliseconds).
- `sjit` (*float*): Source jitter (milliseconds).
- `djit` (*float*): Destination jitter (milliseconds).
- `swin` (*integer*): Source TCP window advertisement value.
- `stcpb` (*integer*): Source TCP base sequence number.
- `dtcpb` (*integer*): Destination TCP base sequence number.
- `dwin` (*integer*): Destination TCP window advertisement value.
- `tcprtt` (*float*): TCP round-trip time (`synack` + `ackdat`).
- `synack` (*float*): TCP SYN to SYN-ACK setup time.
- `ackdat` (*float*): TCP SYN-ACK to ACK completion time.
- `smean` (*integer*): Mean packet size transmitted by source.
- `dmean` (*integer*): Mean packet size transmitted by destination.
- `trans_depth` (*integer*): Pipelined HTTP request/response transaction depth.
- `response_body_len` (*integer*): Actual uncompressed content length of HTTP response.

### 3.4 Connection and Count Features (100-Connection Sliding Window)
- `ct_srv_src`: Connections with same service and source address in past 100 flows.
- `ct_state_ttl`: Connections with identical state and TTL combination.
- `ct_dst_ltm`: Connections to same destination address in past 100 flows.
- `ct_src_dport_ltm`: Connections from same source to same destination port.
- `ct_dst_sport_ltm`: Connections from same destination to same source port.
- `ct_dst_src_ltm`: Connections between same source and destination address pair.
- `is_ftp_login` (*binary*): 1 if FTP session logged in with valid credentials, 0 otherwise.
- `ct_ftp_cmd`: Count of FTP commands issued in the session.
- `ct_flw_http_mthd`: Count of HTTP methods (`GET`, `POST`, etc.) in flow.
- `ct_src_ltm`: Connections from same source address in past 100 flows.
- `ct_srv_dst`: Connections with same service to same destination address.
- `is_sm_ips_ports` (*binary*): 1 if source and destination IP and ports are equal (LAND attack indicator).

### 3.5 Target Labels
- `label` (*binary integer*): `0` for Benign/Normal traffic; `1` for Attack/Malicious traffic.
- `attack_cat` (*nominal*): Threat category (`Normal`, `Fuzzers`, `Analysis`, `Backdoors`, `DoS`, `Exploits`, `Generic`, `Reconnaissance`, `Shellcode`, `Worms`).

---

## 4. Synthetic Host Event Telemetry Schema

The deterministic rule engine evaluates structured host event telemetry from `data/sample/synthetic_security_events.csv`:

| Field Name | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `event_id` | String | Unique event identifier | `EVT-0001` |
| `timestamp` | String | ISO 8601 UTC timestamp | `2026-10-13T10:00:00Z` |
| `source_ip` | String | Originating host IP address | `192.168.1.105` |
| `destination_ip` | String | Target server IP address | `10.0.0.5` |
| `destination_port` | Integer | Target port number | `22`, `80`, `443` |
| `failed_logins` | Integer | Sequential failed authentication attempts | `0`, `12` |
| `byte_count` | Integer | Total payload bytes transferred | `15,482,900` |
| `packet_count` | Integer | Total network packets in session | `2`, `140` |
| `user_agent` | String | HTTP User-Agent header string | `Mozilla/5.0...`, `sqlmap/1.6` |

---

## 5. Official Academic Citation

When citing or referencing the benchmark dataset:

```bibtex
@inproceedings{moustafa2015unsw,
  title={UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)},
  author={Moustafa, Nour and Slay, Jill},
  booktitle={2015 Military Communications and Information Systems Conference (MilCIS)},
  pages={1--6},
  year={2015},
  organization={IEEE},
  doi={10.1109/MilCIS.2015.7348942}
}
```
