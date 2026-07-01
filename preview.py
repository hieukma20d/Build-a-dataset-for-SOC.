"""
Pipeline xây dựng tập dữ liệu fine-tuning SOC từ CIC-IDS-2017
Mục tiêu: 350 mẫu chất lượng cao dạng instruction-tuning
Output: soc_dataset_350.jsonl
"""

import pandas as pd
import json
import numpy as np
import os
import glob
from pathlib import Path

# ============================================================
# BƯỚC 1: Load và gộp nhiều file CSV của CIC-IDS-2017
# ============================================================

DATA_DIR = r"E:\dowwload\MachineLearningCSV\MachineLearningCVE"

# Tự động tìm tất cả file CSV trong thư mục
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"[INFO] Tìm thấy {len(csv_files)} file CSV:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# Load và gộp tất cả file
dfs = []
for f in csv_files:
    try:
        df_temp = pd.read_csv(f, low_memory=False)
        print(f"  [OK] {os.path.basename(f)}: {len(df_temp)} dòng")
        dfs.append(df_temp)
    except Exception as e:
        print(f"  [ERR] {os.path.basename(f)}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"\n[INFO] Tổng cộng: {len(df)} dòng sau khi gộp")

# ============================================================
# BƯỚC 2: Làm sạch dữ liệu
# ============================================================

# Chuẩn hóa tên cột (CIC-IDS-2017 có dấu cách đầu tên cột)
df.columns = df.columns.str.strip()

# Xem nhãn thực tế trong dataset
print("\n[INFO] Các nhãn tấn công trong dataset:")
print(df['Label'].value_counts())

# Loại bỏ giá trị vô cực và NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"[INFO] Sau khi loại NaN/Inf: {len(df)} dòng")

# ============================================================
# BƯỚC 3: Mapping nhãn → loại tấn công chuẩn hóa
# ============================================================

# CIC-IDS-2017 dùng nhiều tên biến thể, cần normalize
LABEL_MAP = {
    # DoS/DDoS
    'DoS Hulk':             'DoS',
    'DoS GoldenEye':        'DoS',
    'DoS slowloris':        'DoS',
    'DoS Slowhttptest':     'DoS',
    'DDoS':                 'DDoS',
    # Web Attacks
    'Web Attack \x96 Brute Force': 'WebAttack',
    'Web Attack \x96 XSS':         'WebAttack',
    'Web Attack \x96 Sql Injection':'WebAttack',
    'Web Attack – Brute Force':    'WebAttack',
    'Web Attack – XSS':            'WebAttack',
    'Web Attack – Sql Injection':  'WebAttack',
    # Khác
    'FTP-Patator':          'BruteForce',
    'SSH-Patator':          'BruteForce',
    'Bot':                  'Bot',
    'PortScan':             'PortScan',
    'Infiltration':         'Infiltration',
    'Heartbleed':           'Heartbleed',
    'BENIGN':               'BENIGN',
}

df['AttackType'] = df['Label'].map(LABEL_MAP)

# Loại bỏ nhãn không có trong map (nếu có)
df_clean = df[df['AttackType'].notna()].copy()
print("\n[INFO] Phân bố sau normalize:")
print(df_clean['AttackType'].value_counts())

# ============================================================
# BƯỚC 4: Chọn features quan trọng cho SOC analyst
# ============================================================

# Các cột cần thiết (tên chuẩn CIC-IDS-2017 sau khi strip)
FEATURE_COLS = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Total',
    'Bwd IAT Total',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Active Mean',
    'Active Std',
    'Idle Mean',
    'Idle Std',
    'Destination Port',
    'Protocol',
]

# Chỉ giữ các cột tồn tại trong dataset
available_cols = [c for c in FEATURE_COLS if c in df_clean.columns]
missing_cols = [c for c in FEATURE_COLS if c not in df_clean.columns]
if missing_cols:
    print(f"\n[WARN] Các cột không tìm thấy (sẽ bỏ qua): {missing_cols}")

df_features = df_clean[available_cols + ['AttackType', 'Label']].copy()

# ============================================================
# BƯỚC 5: Sampling chiến lược — 350 mẫu chất lượng
# ============================================================

# Phân bổ mẫu theo mức độ quan trọng với SOC
SAMPLE_QUOTA = {
    'DDoS':         70,   # Phổ biến nhất, cần nhiều mẫu
    'DoS':          60,
    'PortScan':     50,
    'BruteForce':   40,
    'Bot':          35,
    'WebAttack':    35,
    'BENIGN':       40,   # Cần đủ mẫu BENIGN để model không over-detect
    'Infiltration': 10,
    'Heartbleed':   10,
}
# Tổng = 350

samples = []
actual_counts = {}

for attack_type, quota in SAMPLE_QUOTA.items():
    subset = df_features[df_features['AttackType'] == attack_type]
    available = len(subset)

    if available == 0:
        print(f"[WARN] Không có mẫu '{attack_type}' trong dataset, bỏ qua.")
        continue

    n = min(quota, available)
    sampled = subset.sample(n=n, random_state=42)
    samples.append(sampled)
    actual_counts[attack_type] = n
    print(f"[OK] {attack_type}: lấy {n}/{available} mẫu")

final_df = pd.concat(samples, ignore_index=True)
total_samples = len(final_df)
print(f"\n[INFO] Tổng mẫu thu thập: {total_samples}")

# ============================================================
# BƯỚC 6: "Dịch" thông số kỹ thuật → ngôn ngữ tư duy SOC
# ============================================================

def interpret_flags(row):
    """Đọc TCP flags và trả về mô tả kỹ thuật"""
    flags = []
    if row.get('SYN Flag Count', 0) > 0:
        flags.append("SYN")
    if row.get('ACK Flag Count', 0) > 0:
        flags.append("ACK")
    if row.get('FIN Flag Count', 0) > 0:
        flags.append("FIN")
    if row.get('RST Flag Count', 0) > 0:
        flags.append("RST")
    if row.get('PSH Flag Count', 0) > 0:
        flags.append("PSH")
    if row.get('URG Flag Count', 0) > 0:
        flags.append("URG")
    return ", ".join(flags) if flags else "None"

def classify_flow_rate(bps):
    """Phân loại lưu lượng theo ngưỡng SOC"""
    if pd.isna(bps) or bps <= 0:
        return "không xác định"
    elif bps < 1_000:
        return "rất thấp (<1 KB/s)"
    elif bps < 100_000:
        return "thấp (1–100 KB/s)"
    elif bps < 1_000_000:
        return "trung bình (100 KB/s – 1 MB/s)"
    elif bps < 100_000_000:
        return "cao (1–100 MB/s)"
    else:
        return "rất cao (>100 MB/s)"

def classify_duration(microseconds):
    """Phân loại thời gian flow"""
    if pd.isna(microseconds) or microseconds <= 0:
        return "tức thì (<1ms)"
    seconds = microseconds / 1_000_000
    if seconds < 0.001:
        return "tức thì (<1ms)"
    elif seconds < 1:
        return f"ngắn ({seconds*1000:.1f}ms)"
    elif seconds < 60:
        return f"trung bình ({seconds:.1f}s)"
    else:
        return f"dài ({seconds/60:.1f} phút)"

def get_port_context(port):
    """Thêm context cho các port quan trọng"""
    PORT_NAMES = {
        80: "HTTP", 443: "HTTPS", 22: "SSH", 21: "FTP",
        23: "Telnet", 3389: "RDP", 3306: "MySQL", 1433: "MSSQL",
        8080: "HTTP-alt", 445: "SMB", 53: "DNS", 25: "SMTP",
        6379: "Redis", 27017: "MongoDB", 5432: "PostgreSQL",
    }
    try:
        p = int(port)
        return f"{p} ({PORT_NAMES.get(p, 'custom')})"
    except:
        return str(port)

def build_soc_observation(row):
    """
    Xây dựng phần mô tả kỹ thuật của một flow network
    từ góc nhìn của analyst SOC
    """
    duration_str = classify_duration(row.get('Flow Duration', 0))
    fwd_pkts = int(row.get('Total Fwd Packets', 0))
    bwd_pkts = int(row.get('Total Backward Packets', 0))
    total_pkts = fwd_pkts + bwd_pkts
    fwd_bytes = row.get('Total Length of Fwd Packets', 0)
    bwd_bytes = row.get('Total Length of Bwd Packets', 0)
    total_bytes = fwd_bytes + bwd_bytes
    flow_rate_str = classify_flow_rate(row.get('Flow Bytes/s', 0))
    pkt_rate = row.get('Flow Packets/s', 0)
    flags_str = interpret_flags(row)
    dst_port_str = get_port_context(row.get('Destination Port', 0))
    avg_pkt_size = row.get('Average Packet Size', 0)
    syn_count = int(row.get('SYN Flag Count', 0))
    rst_count = int(row.get('RST Flag Count', 0))
    iat_mean = row.get('Flow IAT Mean', 0)
    proto = int(row.get('Protocol', 6))
    proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto, f"Proto-{proto}")

    observation = (
        f"Giao thức: {proto_name} | Cổng đích: {dst_port_str} | "
        f"Thời gian flow: {duration_str} | "
        f"Tổng gói tin: {total_pkts} (Fwd: {fwd_pkts}, Bwd: {bwd_pkts}) | "
        f"Tổng bytes: {total_bytes:.0f} (Fwd: {fwd_bytes:.0f}, Bwd: {bwd_bytes:.0f}) | "
        f"Tốc độ lưu lượng: {flow_rate_str} | "
        f"Tốc độ gói tin: {pkt_rate:.1f} pkt/s | "
        f"Kích thước gói TB: {avg_pkt_size:.1f} bytes | "
        f"TCP Flags: {flags_str} | "
        f"SYN count: {syn_count} | RST count: {rst_count} | "
        f"IAT trung bình: {iat_mean:.1f}µs"
    )
    return observation

# ============================================================
# BƯỚC 7: Tạo instruction/output theo từng loại tấn công
# ============================================================

ATTACK_PROFILES = {
    'DDoS': {
        'threat_level': 'CRITICAL',
        'classification': 'Distributed Denial of Service (DDoS)',
        'description': (
            'Flow thể hiện đặc trưng của tấn công DDoS: lưu lượng cực cao, '
            'tốc độ gói tin bất thường, tỷ lệ Fwd/Bwd mất cân bằng nghiêm trọng. '
            'Nguồn tấn công có thể đến từ botnet phân tán.'
        ),
        'ioc': [
            'Tốc độ gói tin vượt ngưỡng baseline của hệ thống',
            'Tỷ lệ gói tin Fwd >> Bwd (server không kịp phản hồi)',
            'SYN flood hoặc UDP flood không có handshake hoàn chỉnh',
            'IAT (Inter-Arrival Time) rất thấp và đồng đều',
        ],
        'mitre': 'T1498 - Network Denial of Service',
        'actions': [
            'Rate limit ngay lập tức tại firewall/load balancer',
            'Kích hoạt upstream DDoS mitigation (nếu có Cloudflare/Akamai)',
            'Blackhole routing đối với IP nguồn nếu xác định được',
            'Thu thập PCAP trong 5 phút để phân tích signature',
            'Thông báo ISP upstream để lọc tại edge',
        ],
        'escalation': 'Escalate ngay lên SOC Lead nếu băng thông vượt 50% capacity',
    },
    'DoS': {
        'threat_level': 'HIGH',
        'classification': 'Denial of Service (DoS) — Single Source',
        'description': (
            'Flow từ một nguồn duy nhất với các đặc trưng DoS: '
            'khai thác lỗ hổng giao thức (slowloris, GoldenEye, Hulk) '
            'hoặc flood volume để làm cạn tài nguyên server.'
        ),
        'ioc': [
            'Flow duration dài bất thường (slowloris) hoặc volume rất cao (hulk)',
            'Tỷ lệ kết nối dở dang (SYN mà không có ACK)',
            'HTTP request rate cao đến cùng endpoint',
            'Window size bất thường trong TCP header',
        ],
        'mitre': 'T1499 - Endpoint Denial of Service',
        'actions': [
            'Block IP nguồn tại firewall ngay lập tức',
            'Kiểm tra trạng thái kết nối TCP trên server (netstat -an)',
            'Xem xét giới hạn max connection per IP',
            'Kiểm tra log web server để xác định endpoint bị tấn công',
            'Bật SYN cookies nếu phát hiện SYN flood',
        ],
        'escalation': 'Escalate nếu dịch vụ bị ảnh hưởng (response time > 5s)',
    },
    'PortScan': {
        'threat_level': 'MEDIUM',
        'classification': 'Reconnaissance — Port Scanning',
        'description': (
            'Hoạt động quét cổng từ một IP nguồn: kẻ tấn công đang thu thập '
            'thông tin về hạ tầng mạng, xác định các dịch vụ đang chạy '
            'để chuẩn bị cho bước tấn công tiếp theo.'
        ),
        'ioc': [
            'Một IP kết nối đến nhiều cổng khác nhau trong thời gian ngắn',
            'Phần lớn kết nối bị RST/không có phản hồi (closed ports)',
            'Tỷ lệ SYN cao với ACK thấp (SYN scan / stealth scan)',
            'Gói tin rất nhỏ (chỉ gửi SYN, không có payload)',
        ],
        'mitre': 'T1046 - Network Service Discovery',
        'actions': [
            'Ghi nhận IP nguồn vào threat intelligence watchlist',
            'Kiểm tra firewall rules để đảm bảo các cổng nhạy cảm đã đóng',
            'Xem xét triển khai port knocking hoặc fail2ban',
            'Phân tích xem IP này có hoạt động khác không (correlation)',
            'Theo dõi trong 24h để phát hiện escalation thành exploit',
        ],
        'escalation': 'Escalate nếu IP nguồn bắt đầu kết nối đến cổng dịch vụ quan trọng',
    },
    'BruteForce': {
        'threat_level': 'HIGH',
        'classification': 'Credential Brute Force Attack',
        'description': (
            'Tấn công dò mật khẩu bằng cách thử nhiều credential liên tiếp. '
            'Phổ biến nhất là FTP-Patator và SSH-Patator trong CIC-IDS-2017. '
            'Nguy cơ dẫn đến compromise tài khoản hệ thống.'
        ),
        'ioc': [
            'Số lần kết nối đến cổng SSH/FTP từ một IP trong thời gian ngắn',
            'Các kết nối có duration ngắn đồng đều (kết nối → thất bại → ngắt)',
            'Tỷ lệ RST flag cao (server từ chối sau xác thực thất bại)',
            'IAT đồng đều, có thể tự động hóa bằng script/tool',
        ],
        'mitre': 'T1110 - Brute Force',
        'actions': [
            'Block IP nguồn ngay lập tức sau khi vượt ngưỡng (>=5 lần thất bại)',
            'Kiểm tra log xác thực để xác nhận có tài khoản nào bị compromise không',
            'Reset mật khẩu ngay cho các tài khoản bị target',
            'Bật MFA cho SSH/FTP nếu chưa có',
            'Xem xét đổi cổng dịch vụ (SSH từ 22 sang cổng khác)',
        ],
        'escalation': 'CRITICAL nếu phát hiện đăng nhập thành công sau chuỗi thất bại',
    },
    'Bot': {
        'threat_level': 'HIGH',
        'classification': 'Botnet Command & Control (C2) Traffic',
        'description': (
            'Lưu lượng C2 của botnet: máy trong mạng đã bị nhiễm malware và '
            'đang giao tiếp với C2 server của attacker. Có thể được dùng để '
            'DDoS, spam, data exfiltration, hoặc lateral movement.'
        ),
        'ioc': [
            'Kết nối định kỳ đến IP/domain không rõ nguồn gốc (beacon interval)',
            'Lưu lượng ra ngoài bất thường từ workstation/server nội bộ',
            'Encrypted traffic đến cổng không chuẩn',
            'Tỷ lệ bytes outbound >> inbound (upload dữ liệu)',
        ],
        'mitre': 'T1071 - Application Layer Protocol (C2)',
        'actions': [
            'Cô lập ngay máy bị nghi nhiễm khỏi network',
            'Chặn IP/domain C2 tại firewall và DNS',
            'Thu thập memory dump và disk image cho forensic',
            'Quét toàn bộ subnet để tìm máy nhiễm khác (lateral spread)',
            'Báo cáo IOC lên threat intelligence platform',
        ],
        'escalation': 'CRITICAL — kích hoạt Incident Response Plan ngay lập tức',
    },
    'WebAttack': {
        'threat_level': 'HIGH',
        'classification': 'Web Application Attack (SQLi/XSS/BruteForce)',
        'description': (
            'Tấn công vào ứng dụng web: SQL Injection nhằm trích xuất dữ liệu DB, '
            'XSS để đánh cắp session/cookie, hoặc brute force vào form đăng nhập. '
            'Nguy cơ data breach và account takeover.'
        ),
        'ioc': [
            'HTTP requests chứa ký tự đặc biệt: single quote, script tags, UNION SELECT',
            'Payload size bất thường (quá nhỏ hoặc quá lớn so với request thông thường)',
            'Response code 500 (Internal Server Error) liên tiếp',
            'Cùng IP gửi nhiều request đến endpoint login/search/API',
        ],
        'mitre': 'T1190 - Exploit Public-Facing Application',
        'actions': [
            'Xem xét WAF logs để xác định payload cụ thể',
            'Block IP nguồn và tạo WAF rule mới',
            'Kiểm tra DB logs để xác nhận có data exfiltration không',
            'Review code ứng dụng tại endpoint bị tấn công',
            'Kiểm tra tất cả tài khoản đăng nhập trong 24h qua',
        ],
        'escalation': 'Escalate ngay nếu query DB có dấu hiệu thành công (response lớn bất thường)',
    },
    'BENIGN': {
        'threat_level': 'INFO',
        'classification': 'Lưu lượng bình thường (Benign Traffic)',
        'description': (
            'Flow mạng nằm trong baseline bình thường. '
            'Các thông số về tốc độ, tỷ lệ gói tin, TCP flags đều '
            'nằm trong ngưỡng hoạt động thông thường của hệ thống.'
        ),
        'ioc': [],
        'mitre': 'N/A',
        'actions': [
            'Không có hành động cần thiết',
            'Tiếp tục monitoring theo baseline',
        ],
        'escalation': 'Không cần escalate',
    },
    'Infiltration': {
        'threat_level': 'CRITICAL',
        'classification': 'Network Infiltration — Post-Compromise Activity',
        'description': (
            'Dấu hiệu của attacker đã xâm nhập vào mạng nội bộ và đang thực hiện '
            'lateral movement hoặc data exfiltration. Giai đoạn nguy hiểm nhất '
            'trong kill chain.'
        ),
        'ioc': [
            'Lưu lượng bất thường giữa các máy nội bộ (East-West traffic)',
            'Kết nối từ server đến workstation (reverse connection)',
            'Upload dữ liệu lớn ra ngoài vào giờ thấp điểm',
            'Sử dụng giao thức tunneling (DNS tunneling, ICMP tunneling)',
        ],
        'mitre': 'T1041 - Exfiltration Over C2 Channel',
        'actions': [
            'Kích hoạt Incident Response Plan ngay lập tức',
            'Cô lập toàn bộ segment mạng bị ảnh hưởng',
            'Thu thập evidence: PCAP, logs, memory dump',
            'Notify CISO và Legal team (có thể liên quan PDPA/breach notification)',
            'Engage Digital Forensics & Incident Response (DFIR) team',
        ],
        'escalation': 'CRITICAL — Toàn bộ incident response team phải được kích hoạt',
    },
    'Heartbleed': {
        'threat_level': 'CRITICAL',
        'classification': 'Heartbleed (CVE-2014-0160) — OpenSSL Vulnerability Exploit',
        'description': (
            'Khai thác lỗ hổng Heartbleed trong OpenSSL: attacker gửi malformed '
            'heartbeat request để đọc 64KB bộ nhớ server, có thể thu được '
            'private key SSL, session token, mật khẩu người dùng.'
        ),
        'ioc': [
            'Request HTTPS với TLS heartbeat record bất thường (size mismatch)',
            'Server phản hồi với payload lớn hơn mong đợi',
            'Cổng 443 nhận traffic pattern không chuẩn của SSL handshake',
            'Repeated connections từ cùng IP với pattern heartbeat',
        ],
        'mitre': 'T1190 - Exploit Public-Facing Application (CVE-2014-0160)',
        'actions': [
            'KHẨN CẤP: Patch OpenSSL lên version không bị ảnh hưởng',
            'Thu hồi và cấp lại toàn bộ SSL certificate',
            'Vô hiệu hóa tất cả session token hiện tại (force logout)',
            'Kiểm tra logs để xác định data nào đã bị lộ',
            'Notify người dùng thay đổi mật khẩu',
        ],
        'escalation': 'CRITICAL — xử lý ngay, mọi giờ trong ngày',
    },
}

def build_instruction(row):
    """Tạo câu hỏi/instruction cho model"""
    observation = build_soc_observation(row)
    attack_type = row['AttackType']
    
    # Thêm context ngẫu nhiên để đa dạng hóa instruction
    templates = [
        f"Phân tích flow mạng sau và đưa ra đánh giá bảo mật:\n{observation}",
        f"Với tư cách chuyên gia SOC, hãy phân loại và xử lý luồng mạng:\n{observation}",
        f"Hệ thống SIEM cảnh báo về flow mạng sau. Hãy phân tích và đề xuất hành động:\n{observation}",
        f"Đánh giá mức độ nguy hiểm của flow mạng này và đề xuất biện pháp ứng phó:\n{observation}",
    ]
    
    import random
    random.seed(int(row.name) % 1000)  # Reproducible
    return random.choice(templates)

def build_output(row):
    """Tạo response SOC expert cho một flow"""
    attack_type = row['AttackType']
    profile = ATTACK_PROFILES.get(attack_type, ATTACK_PROFILES['BENIGN'])
    observation = build_soc_observation(row)
    
    # Build IOC string
    ioc_str = ""
    if profile['ioc']:
        ioc_lines = "\n".join([f"  - {ioc}" for ioc in profile['ioc']])
        ioc_str = f"\n\nCác Indicator of Compromise (IoC) phù hợp:\n{ioc_lines}"
    
    # Build actions string
    action_lines = "\n".join([f"  {i+1}. {act}" for i, act in enumerate(profile['actions'])])
    
    output = (
        f"**PHÂN TÍCH FLOW MẠNG**\n\n"
        f"Mức độ đe dọa: {profile['threat_level']}\n"
        f"Phân loại: {profile['classification']}\n\n"
        f"**Nhận định:**\n{profile['description']}"
        f"{ioc_str}\n\n"
        f"**Khung MITRE ATT&CK:** {profile['mitre']}\n\n"
        f"**Hành động đề xuất:**\n{action_lines}\n\n"
        f"**Escalation Policy:** {profile['escalation']}"
    )
    return output

# ============================================================
# BƯỚC 8: Tạo tập dữ liệu JSON Lines
# ============================================================

print("\n[INFO] Đang tạo instruction-output pairs...")

dataset = []

for idx, row in final_df.iterrows():
    try:
        instruction = build_instruction(row)
        output = build_output(row)
        
        sample = {
            "instruction": instruction,
            "output": output,
            "metadata": {
                "attack_type": row['AttackType'],
                "original_label": row['Label'],
                "threat_level": ATTACK_PROFILES.get(row['AttackType'], {}).get('threat_level', 'UNKNOWN'),
                "source": "CIC-IDS-2017",
            }
        }
        dataset.append(sample)
    except Exception as e:
        print(f"[WARN] Lỗi tại index {idx}: {e}")
        continue

print(f"[INFO] Tạo thành công {len(dataset)} mẫu")

# ============================================================
# BƯỚC 9: Xuất file JSONL
# ============================================================

OUTPUT_FILE = "soc_dataset_350.jsonl"

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for sample in dataset:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\n[DONE] Đã lưu tập dữ liệu vào: {OUTPUT_FILE}")
print(f"[DONE] Tổng mẫu: {len(dataset)}")
print("\n[INFO] Thống kê phân bổ cuối cùng:")
from collections import Counter
type_count = Counter(s['metadata']['attack_type'] for s in dataset)
for k, v in sorted(type_count.items(), key=lambda x: -x[1]):
    print(f"  {k:20s}: {v} mẫu")

# ============================================================
# BƯỚC 10 (TÙY CHỌN): Xuất CSV để kiểm tra bằng Excel
# ============================================================

df_export = pd.DataFrame([{
    'attack_type': s['metadata']['attack_type'],
    'threat_level': s['metadata']['threat_level'],
    'instruction_preview': s['instruction'][:120] + '...',
    'output_preview': s['output'][:200] + '...',
} for s in dataset])

df_export.to_csv("soc_dataset_preview.csv", index=False, encoding='utf-8-sig')
print("\n[DONE] File preview đã lưu vào: soc_dataset_preview.csv")
print("       (Mở bằng Excel để kiểm tra nhanh nội dung)")