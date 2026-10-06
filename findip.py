import subprocess
import platform
import socket
import argparse
import re
from concurrent.futures import ThreadPoolExecutor

def ping(ip):
    param = "-n" if platform.system().lower() == "windows" else "-c"
    cmd = ["ping", param, "1", "-W", "1", ip]
    return subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode == 0

def get_hostname_dns(ip):
    """일반 DNS 역방향 조회"""
    try:
        hostname = socket.gethostbyaddr(ip)[0]
        return hostname
    except:
        return None

def get_hostname_mdns(ip):
    """mDNS (avahi) 조회 - 라즈베리파이용"""
    try:
        # avahi-resolve 사용
        result = subprocess.run(
            ["avahi-resolve", "-a", ip],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # 출력 형식: "192.168.0.179  raspberrypi.local"
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return parts[1].replace('.local', '')
        return None
    except:
        return None

def get_hostname_nmblookup(ip):
    """NetBIOS 이름 조회 (Windows/Samba)"""
    try:
        result = subprocess.run(
            ["nmblookup", "-A", ip],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # NetBIOS 이름 파싱
            for line in result.stdout.split('\n'):
                if '<00>' in line and 'GROUP' not in line:
                    name = line.split()[0].strip()
                    if name and name != ip:
                        return name
        return None
    except:
        return None

def get_hostname(ip):
    """모든 방법으로 hostname 조회 시도"""
    # 1. DNS 조회
    hostname = get_hostname_dns(ip)
    if hostname:
        return hostname
    
    # 2. mDNS 조회 (라즈베리파이)
    hostname = get_hostname_mdns(ip)
    if hostname:
        return hostname
    
    # 3. NetBIOS 조회
    hostname = get_hostname_nmblookup(ip)
    if hostname:
        return hostname
    
    return None

def get_mac_address(ip):
    """ARP 테이블에서 MAC 주소 가져오기"""
    try:
        if platform.system().lower() == "windows":
            cmd = ["arp", "-a", ip]
        else:
            cmd = ["arp", "-n", ip]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )
        
        mac_pattern = r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})'
        match = re.search(mac_pattern, result.stdout)
        
        if match:
            return match.group(0).upper()
        return None
    except:
        return None

def get_vendor_from_mac(mac):
    """MAC 주소로 제조사 확인"""
    if not mac:
        return None
    
    oui_db = {
        'B8:27:EB': 'Raspberry Pi Foundation',
        'DC:A6:32': 'Raspberry Pi Trading',
        'E4:5F:01': 'Raspberry Pi Trading',
        '28:CD:C1': 'Raspberry Pi Trading',
        'D8:3A:DD': 'Raspberry Pi Trading',
        '2C:CF:67': 'Raspberry Pi Trading',
        'B0:7D:64': 'Espressif (ESP32)',
        '24:0A:C4': 'Espressif (ESP32)',
        '30:AE:A4': 'Espressif (ESP32)',
        '84:CC:A8': 'Espressif (ESP32)',
        'C8:2B:96': 'Khadas',
    }
    
    mac_prefix = mac[:8].upper()
    return oui_db.get(mac_prefix, None)

def check_host(ip):
    """호스트 정보 수집"""
    if ping(ip):
        hostname = get_hostname(ip)
        mac = get_mac_address(ip)
        vendor = get_vendor_from_mac(mac)
        return (ip, hostname, mac, vendor)
    return None

def scan(base):
    ips = [base + str(i) for i in range(1, 255)]
    
    print(f"Scanning {base}0/24 ...\n")
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(check_host, ips))
    
    alive_hosts = [r for r in results if r is not None]
    
    # 결과 출력
    print(f"{'IP Address':<17} {'Hostname':<25} {'MAC Address':<20} {'Vendor'}")
    print("-" * 95)
    
    for ip, hostname, mac, vendor in alive_hosts:
        hostname_str = hostname if hostname else "unknown"
        mac_str = mac if mac else "N/A"
        vendor_str = vendor if vendor else ""
        
        # Raspberry Pi만 강조 표시
        if vendor and 'Raspberry Pi' in vendor:
            print(f"{ip:<17} {hostname_str:<25} {mac_str:<20} [RPI] {vendor_str}")
        else:
            print(f"{ip:<17} {hostname_str:<25} {mac_str:<20} {vendor_str}")
    
    # Raspberry Pi 요약
    rpi_hosts = [h for h in alive_hosts if h[3] and 'Raspberry Pi' in h[3]]
    if rpi_hosts:
        print(f"\n{'='*95}")
        print(f"Found {len(rpi_hosts)} Raspberry Pi device(s):")
        for ip, hostname, mac, vendor in rpi_hosts:
            hostname_str = hostname if hostname else "unknown"
            print(f"  • {hostname_str:<25} → {ip:<17} ({mac})")
    
    print(f"\nTotal alive hosts: {len(alive_hosts)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced LAN scanner with mDNS support")
    parser.add_argument(
        "base",
        help="Base IP (example: 192.168.0.)"
    )
    args = parser.parse_args()
    
    scan(args.base)
